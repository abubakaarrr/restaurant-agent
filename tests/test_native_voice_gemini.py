"""Gemini transport and restaurant authorization boundary regressions."""
import asyncio
import base64
from dataclasses import replace
from datetime import date
import json
import os
import time
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import pytest

from app.native_voice.adapter import InterruptionController, NativeVoiceAdapter
from app.native_voice.contracts import OrderState
from app.native_voice.gemini_live import GeminiLiveSession, canonical_available_menu_item, extract_reservation_slots, native_setup, party_size_correction, reservation_readback_matches
from app.native_voice.streaming import menu_suggestions, restaurant_information, state_view
from app.native_voice.tools import ToolOutcome
from app.native_voice.tools import ToolBridge
from app.call_memory import clear_call_memory
from app.pending_confirmation import (
    ACTION_CONFIRM_ORDER, ACTION_CREATE_BOOKING, ACTION_UPDATE_CONFIRMED_BOOKING,
    begin_caller_turn, get_pending_confirmation, register_pending_confirmation,
    release_pending_confirmation,
)


def make_session():
    events = []
    async def emit(event): events.append(event)
    domain = SimpleNamespace(state=OrderState(), session_id='gemini-test', interruptions=InterruptionController(), _outcomes=[], _completed_turn=None,
        _release_pending_readbacks=AsyncMock(), _run_tool=AsyncMock(), _persist_native_confirmation_state=AsyncMock())
    socket = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
    session = GeminiLiveSession(domain=domain, socket=socket, send=emit, pace='natural')
    return session, events


def test_one_native_model_and_blocking_tools():
    session, _ = make_session()
    config = native_setup(session.domain, 'natural')['setup']
    assert config['model'] == 'models/gemini-3.8-live'
    assert config['generationConfig']['responseModalities'] == ['AUDIO']
    assert config['inputAudioTranscription'] == {}
    assert 'openai' not in json.dumps(config).lower()
    assert all(tool['behavior'] == 'BLOCKING' for tool in config['tools'][0]['functionDeclarations'])


def test_pcm_exact_bytes_and_native_input_format():
    async def run():
        session, _ = make_session()
        pcm = b'\x00\x00\xff\x7f\x00\x80'
        await session.append_audio(pcm)
        data = json.loads(session.socket.send.call_args.args[0])['realtimeInput']['audio']
        assert data['mimeType'] == 'audio/pcm;rate=16000'
        assert base64.b64decode(data['data']) == pcm
        for invalid in (b'', b'\0', bytes(32002)):
            try: await session.append_audio(invalid)
            except ValueError: pass
            else: raise AssertionError('invalid PCM accepted')
    asyncio.run(run())


def test_names_must_match_whole_caller_tokens():
    session, _ = make_session()
    session.input_history = ['My name is Abubakar.', 'My colleague is Joanne.']
    assert session.name_supported({'customer_name':'Abubakar'})
    assert not session.name_supported({'customer_name':'Jacob'})
    assert not session.name_supported({'customer_name':'Ann'})


def test_spelled_name_is_supported_but_phone_is_never_a_name():
    session, _ = make_session()
    session.input_text = 'My name is A B U B A K A R.'
    assert session.name_supported({'name':'Abubakar'})
    assert not session.name_supported({'name':'2233314'})


def test_reservation_slots_survive_a_later_name_clarification():
    slots = extract_reservation_slots(
        'Please reserve a table for four this Saturday at 7:30 p.m. '
        'My callback number is 503-555-0123.',
        today=date(2026, 9, 23),
    )
    assert slots == {
        'phone':'5035550123', 'party_size':4,
        'time':'19:30', 'date':'2026-09-26',
    }


def test_relative_guest_change_uses_current_party_size():
    assert party_size_correction('Can I add one more guest to my reservation?', 4) == 5
    assert party_size_correction('One fewer person will join us.', 4) == 3
    assert extract_reservation_slots('We will be five people instead of four.', today=date(2026, 9, 23))['party_size'] == 5


def test_contextual_guest_notes_and_missing_high_chair_quantity():
    note, question = GeminiLiveSession.contextual_note('vegetarian', 'One of my friends is vegetarian.')
    assert note == 'One guest is vegetarian.' and question is None
    _, question = GeminiLiveSession.contextual_note('high chair', 'Can we have a high chair for my child?')
    assert 'how many high chairs' in question
    note, question = GeminiLiveSession.contextual_note('two high chairs', 'We need two high chairs for the children.')
    assert '2 high chair' in note and question is None


def test_unverified_play_area_is_not_claimed():
    answer = restaurant_information('Do you have a play area for children?')
    assert "can't confirm" in answer
    assert 'play area' in answer


def test_menu_suggestions_only_use_live_available_dietary_items():
    async def run():
        service = SimpleNamespace(list_menu=AsyncMock(return_value={'items': [
            {'name':'Wild Mushroom Grain Bowl','item_id':'menu.veg.mushroom-grain-bowl','category':'vegetarian_vegan','dietary_tags':['vegan','vegetarian'],'price':20,'available':True},
            {'name':'Unavailable Ravioli','item_id':'menu.seasonal.ravioli','category':'vegetarian_vegan','dietary_tags':['vegetarian'],'price':24,'available':False},
            {'name':'Chicken Caesar Salad','item_id':'menu.salad.chicken-caesar','category':'salads','dietary_tags':[],'price':19,'available':True},
        ]}))
        result = await menu_suggestions(service, 'vegetarian')
        assert [item['name'] for item in result['items']] == ['Wild Mushroom Grain Bowl']
        assert 'Unavailable Ravioli' not in result['answer']
        service.list_menu.assert_awaited_once_with(available_only=True)
    asyncio.run(run())


def test_confirmed_order_item_singularization_requires_unique_live_speech_match():
    items = [
        {'name':'Market Greens','aliases':['green salad'],'available':True},
        {'name':'Chicken Caesar Salad','available':True},
    ]
    assert canonical_available_menu_item(items, 'Market Green salad', 'Please add one Market Green salad.')['name'] == 'Market Greens'
    assert canonical_available_menu_item(items, 'Market Greens', 'Please add a Caesar salad.') is None


def test_confirmed_booking_correction_uses_current_id_and_checks_new_party():
    async def run():
        session, _ = make_session()
        session.finalize_input = AsyncMock()
        session.input_text = 'Please add one more guest to my reservation.'
        session.domain._native_service = SimpleNamespace(load_call_state=AsyncMock(return_value={
            'state': {'reservation_draft': {
                'booking_id': 235, 'status':'confirmed', 'customer_name':'Abubakar',
                'date':'2026-09-26', 'time':'19:30', 'party_size':4,
            }}
        }))
        availability = ToolOutcome(name='check_table_availability', call_id='change:availability', arguments={},
            result={'available':True}, success=True, readback_verified=True)
        proposed = ToolOutcome(name='update_confirmed_booking', call_id='change', arguments={},
            result={'proposed':{'booking_id':235,'party_size':5},'pending_confirmation_hash':'digest'},
            success=True, readback_verified=True, pending=True,
            confirmation_text='Booking reference 235 would be updated to party size 5. Would you like me to apply these changes?')
        session.domain._run_tool.side_effect = [availability, proposed]
        session.domain._with_confirmation = lambda outcome: outcome
        session.domain._persist_committed_outcome = AsyncMock(return_value=proposed)
        session.domain._sync_order_memory = AsyncMock(return_value=None)
        session.domain._model_tool_output = lambda outcome: {'status':'pending_confirmation'}
        result = await session.execute_function({'id':'change','name':'update_reservation_draft','args':{'party_size':5}},0)
        assert result['readback_text'] == proposed.confirmation_text
        calls = session.domain._run_tool.await_args_list
        assert calls[0].args[1] == 'check_table_availability'
        assert calls[0].args[2]['party_size'] == 5
        assert calls[1].args[1] == 'update_confirmed_booking'
        assert calls[1].args[2]['booking_id'] == 235
        assert calls[1].args[2]['caller_confirmed'] is False
        assert session.pending_booking_update['party_size'] == 5
    asyncio.run(run())


def test_conflicting_relative_and_explicit_party_change_requires_clarification():
    async def run():
        session, _ = make_session()
        session.finalize_input = AsyncMock()
        session.input_text = 'Add one more guest; we will be five people instead of four.'
        session.current_reservation_slots = {'party_size': 5}
        session.domain._native_service = SimpleNamespace(load_call_state=AsyncMock(return_value={
            'state': {'reservation_draft': {'booking_id': 237, 'status':'confirmed',
                                            'date':'2026-09-25', 'time':'19:00', 'party_size':2}}
        }))
        result = await session.execute_function({'id':'change','name':'update_confirmed_booking','args':{'party_size':5}},0)
        assert result['error'] == 'party_size_conflict'
        session.domain._run_tool.assert_not_called()
    asyncio.run(run())


def test_booking_amendment_readback_names_only_requested_change():
    outcome = ToolOutcome(
        name='update_confirmed_booking', call_id='change', arguments={'booking_id':237, 'party_size':3},
        result={'proposed': {'booking_id':237, 'date':'2026-09-25', 'time':'19:00',
                             'party_size':3, 'customer_name':'Jordan Test', 'dietary':''}},
        success=False, pending=True,
    )
    sentence = NativeVoiceAdapter._confirmation_sentence(outcome)
    assert sentence == 'For booking reference 237, I can change the party size to 3 guests. Should I make that change?'
    assert 'dietary' not in sentence


def test_confirmed_preorder_amendment_waits_for_later_yes():
    async def run():
        session, _ = make_session()
        session.finalize_input = AsyncMock()
        session.domain.state = replace(session.domain.state, status='confirmed')
        session.domain._native_service = SimpleNamespace(list_menu=AsyncMock(return_value={'items':[
            {'name':'Rosemary Fries', 'available':True}
        ]}))
        session.apply_order_changes = AsyncMock(return_value=({'status':'completed'}, None))
        session.input_text = 'Please add one Rosemary Fries to my confirmed pre-order.'
        proposal = {'actions':[{'name':'add_order_item', 'arguments':{'item_name':'Rosemary Fries', 'quantity':1}}],
                    'expected_order_revision':state_view(session.domain.state)['order_revision'], 'retain_order_item_ids':[]}
        pending = await session.execute_function({'id':'proposal', 'name':'apply_order_changes', 'args':proposal}, 0)
        assert pending['status'] == 'needs_confirmation'
        session.apply_order_changes.assert_not_called()
        session.input_text = 'Yes, confirm that.'
        result = await session.execute_function({'id':'approval', 'name':'apply_order_changes', 'args':{}}, 0)
        assert result['status'] == 'completed'
        approved = session.apply_order_changes.await_args.args[0]
        assert approved['actions'][0]['arguments']['caller_confirmed'] is True
        assert approved['actions'][0]['arguments']['item_name'] == 'Rosemary Fries'
    asyncio.run(run())


def test_confirmed_preorder_drops_redundant_fulfillment_before_proposal():
    async def run():
        session, _ = make_session()
        session.finalize_input = AsyncMock()
        session.domain.state = replace(session.domain.state, status='confirmed', fulfillment='dine_in')
        session.domain._native_service = SimpleNamespace(list_menu=AsyncMock(return_value={'items':[
            {'name':'Market Greens', 'available':True}
        ]}))
        session.input_text = 'Please add Market Greens to my confirmed pre-order.'
        proposal = {'actions':[
            {'name':'add_order_item', 'arguments':{'item_name':'Market Greens', 'quantity':1}},
            {'name':'set_order_fulfillment', 'arguments':{'fulfillment_type':'dine_in'}},
        ], 'expected_order_revision':state_view(session.domain.state)['order_revision'], 'retain_order_item_ids':[]}
        pending = await session.execute_function({'id':'proposal', 'name':'apply_order_changes', 'args':proposal}, 0)
        assert pending['status'] == 'needs_confirmation'
        assert [action['name'] for action in session.pending_order_amendment['actions']] == ['add_order_item']
        session.domain._run_tool.assert_not_called()
    asyncio.run(run())


def test_new_dine_in_preorder_defers_fulfillment_until_first_item():
    async def run():
        session, _ = make_session()
        session.finalize_input = AsyncMock()
        session.input_text = 'Add one Chicken Caesar Salad as a dine-in pre-order for my reservation.'
        session.domain._native_service = SimpleNamespace(load_call_state=AsyncMock(return_value={
            'state': {'reservation_draft': {'booking_id':243, 'status':'confirmed'}}
        }))
        session.apply_order_changes = AsyncMock(return_value=({'status':'completed'}, None))
        deferred = await session.execute_function({'id':'fulfillment', 'name':'apply_order_changes',
            'args':{'actions':[{'name':'set_order_fulfillment', 'arguments':{'fulfillment_type':'dine_in',
                                                                           'fulfillment_at':'2026-09-25T19:00:00-07:00'}}]}}, 0)
        assert deferred['status'] == 'deferred'
        session.apply_order_changes.assert_not_called()
        await session.execute_function({'id':'item', 'name':'apply_order_changes',
            'args':{'actions':[{'name':'add_order_item', 'arguments':{'item_name':'Chicken Caesar Salad', 'quantity':1}}]}}, 0)
        actions = session.apply_order_changes.await_args.args[0]['actions']
        assert [action['name'] for action in actions] == ['add_order_item', 'set_order_fulfillment']
        assert actions[1]['arguments'] == {'fulfillment_type':'dine_in'}
    asyncio.run(run())


def test_short_or_unsupported_phone_is_rejected_before_domain_write():
    async def run():
        session, _ = make_session()
        session.input_text = 'My phone is two two three, three three one four.'
        problem = await session.validate_identity_fields({'phone':'2233314'}, confirmation_only=False)
        assert problem['error'] == 'invalid_phone_field'
    asyncio.run(run())


def test_explicit_draft_name_correction_needs_no_second_identity_confirmation():
    async def run():
        session, _ = make_session()
        service = SimpleNamespace(load_call_state=AsyncMock(return_value={
            'state': {'reservation_draft': {'customer_name':'Abubakar'}}
        }))
        session.domain._native_service = service
        session.input_text = 'Actually, correct the name to Alex.'
        problem = await session.validate_identity_fields({'name':'Alex'}, confirmation_only=False)
        assert problem is None
        assert session.reservation_buffer['name'] == 'Alex'
    asyncio.run(run())


def test_explicit_under_name_can_be_saved_in_unconfirmed_draft():
    async def run():
        session, _ = make_session()
        session.input_text = 'Please book a table under Abovakar.'
        problem = await session.validate_identity_fields({'name':'Abovakar'}, confirmation_only=False)
        assert problem is None
        session.input_text = 'My name is Abubakar, spelled A B U B A K A R.'
        assert await session.validate_identity_fields({'name':'Abubakar'}, confirmation_only=False) is None
    asyncio.run(run())


def test_booking_yes_uses_saved_readback_payload_not_model_reconstruction():
    async def run():
        session, _ = make_session()
        session.finalize_input = AsyncMock()
        session.input_text = 'Yes, please confirm that.'
        session.pending_booking_creation = {'name':'Jordan Test', 'phone':'+15035550124',
                                            'date':'2026-09-25', 'time':'19:00', 'party_size':2, 'notes':''}
        session.domain._native_service = SimpleNamespace(load_call_state=AsyncMock(return_value={
            'state': {'reservation_draft': {'customer_name':'Jordan Test', 'customer_phone':'+15035550124'}}
        }))
        failed = ToolOutcome(name='create_booking', call_id='confirm', arguments={}, result={}, success=False, error='fixture')
        session.domain._run_tool.return_value = failed
        session.domain._with_confirmation = lambda outcome: outcome
        session.domain._sync_order_memory = AsyncMock(return_value=None)
        session.domain._model_tool_output = lambda outcome: {'status':'failed'}
        session.speak = AsyncMock()
        with patch.dict('os.environ', {'OPENAI_API_KEY':'test'}), patch('openai.AsyncOpenAI', return_value=SimpleNamespace(close=AsyncMock())):
            await session.execute_function({'id':'confirm', 'name':'get_reservation_draft',
                                            'args':{'name':'Wrong Name', 'date':'2026-09-26', 'party_size':5}}, 0)
        assert session.domain._run_tool.await_args.args[1] == 'create_booking'
        sent = session.domain._run_tool.await_args.args[2]
        assert sent['name'] == 'Jordan Test'
        assert sent['date'] == '2026-09-25'
        assert sent['party_size'] == 2
    asyncio.run(run())


def test_booking_approval_executes_without_waiting_for_model_tool_choice():
    async def run():
        session, _ = make_session()
        session.input_id = 'approval-turn'
        clear_call_memory(session.domain.session_id)
        begin_caller_turn(session.domain.session_id, 'Please read back my reservation')
        digest = register_pending_confirmation(session.domain.session_id, ACTION_CREATE_BOOKING, {
            'customer_name':'Jordan Test', 'customer_phone':'+15035550124',
            'date':'2026-09-25', 'time':'19:00', 'party_size':2, 'notes':''})
        assert release_pending_confirmation(session.domain.session_id, ACTION_CREATE_BOOKING, digest)
        begin_caller_turn(session.domain.session_id, 'Yes, please confirm that')
        confirmed = ToolOutcome(name='create_booking', call_id='approval-turn:confirm', arguments={},
                                result={'booking_id':240}, success=True, readback_verified=True,
                                confirmation_text='Your reservation is confirmed.')
        session.domain._run_tool.return_value = confirmed
        session.domain._with_confirmation = lambda outcome: outcome
        session.domain._persist_committed_outcome = AsyncMock(return_value=confirmed)
        session.domain._sync_order_memory = AsyncMock(return_value=None)
        session.domain._model_tool_output = lambda outcome: {'status':'completed'}
        session.speak = AsyncMock()
        fake_client = SimpleNamespace(close=AsyncMock())
        with patch.dict('os.environ', {'OPENAI_API_KEY':'test-key'}), patch('openai.AsyncOpenAI', return_value=fake_client):
            await session.confirm_pending_booking()
        assert session.domain._run_tool.await_args.args[1] == 'create_booking'
        assert session.domain._run_tool.await_args.args[2]['party_size'] == 2
        assert session.speak.await_args.args[0] == 'Your reservation is confirmed.'
        assert session.block_output
    asyncio.run(run())


def test_real_call_order_approval_executes_without_model_tool_choice():
    async def run():
        session, _ = make_session()
        clear_call_memory(session.domain.session_id)
        begin_caller_turn(session.domain.session_id, 'Please read the order back')
        digest = register_pending_confirmation(session.domain.session_id, ACTION_CONFIRM_ORDER,
                                               {'order_id':106, 'draft_version':3, 'booking_id':247})
        assert release_pending_confirmation(session.domain.session_id, ACTION_CONFIRM_ORDER, digest)
        session.input_id = 'approval-turn'
        session.input_text = 'Ya ya.'
        session.input_ready.set()
        session.input_updated = time.monotonic() - 1
        session.domain.turns = SimpleNamespace(start=lambda _: None)
        async def finalize(text, *, generation):
            begin_caller_turn(session.domain.session_id, text)
        session.domain._finalize_caller_turn = finalize
        confirmed = ToolOutcome(name='confirm_order', call_id='approval-turn:confirm_order', arguments={},
                                result={'order_id':106, 'status':'confirmed'}, success=True,
                                readback_verified=True, confirmation_text='Your pre-order is confirmed.')
        session.domain._run_tool.return_value = confirmed
        session.domain._with_confirmation = lambda outcome: outcome
        session.domain._persist_committed_outcome = AsyncMock(return_value=confirmed)
        session.domain._sync_order_memory = AsyncMock(return_value=None)
        session.domain._model_tool_output = lambda outcome: {'status':'completed'}
        session.speak = AsyncMock()
        fake_client = SimpleNamespace(close=AsyncMock())
        with patch.dict('os.environ', {'OPENAI_API_KEY':'test-key'}), patch('openai.AsyncOpenAI', return_value=fake_client):
            await session.finalize_input()
        assert session.domain._run_tool.await_args.args[1] == 'confirm_order'
        sent = session.domain._run_tool.await_args.args[2]
        assert sent['expected_draft_version'] == 3
        assert sent['caller_approved_full_readback'] is True
        assert session.speak.await_args.args[0] == 'Your pre-order is confirmed.'
        assert session.block_output
    asyncio.run(run())


def test_real_call_booking_change_approval_updates_existing_booking():
    async def run():
        session, _ = make_session()
        clear_call_memory(session.domain.session_id)
        begin_caller_turn(session.domain.session_id, 'Please read the proposed Sunday change')
        digest = register_pending_confirmation(session.domain.session_id, ACTION_UPDATE_CONFIRMED_BOOKING, {
            'booking_id':247, 'date':'2026-09-27', 'time':'18:30',
            'party_size':7, 'customer_name':'Hamza', 'preferred_location':'main'})
        assert release_pending_confirmation(session.domain.session_id, ACTION_UPDATE_CONFIRMED_BOOKING, digest)
        session.input_id = 'approval-turn'
        session.input_text = 'Yes, make that change.'
        session.input_ready.set()
        session.input_updated = time.monotonic() - 1
        session.domain.turns = SimpleNamespace(start=lambda _: None)
        async def finalize(text, *, generation):
            begin_caller_turn(session.domain.session_id, text)
        session.domain._finalize_caller_turn = finalize
        updated = ToolOutcome(name='update_confirmed_booking', call_id='approval-turn:update_confirmed_booking',
                              arguments={}, result={'booking_id':247, 'status':'confirmed'}, success=True,
                              readback_verified=True, confirmation_text='Your reservation was updated.')
        session.domain._run_tool.return_value = updated
        session.domain._with_confirmation = lambda outcome: outcome
        session.domain._persist_committed_outcome = AsyncMock(return_value=updated)
        session.domain._sync_order_memory = AsyncMock(return_value=None)
        session.domain._model_tool_output = lambda outcome: {'status':'completed'}
        session.speak = AsyncMock()
        fake_client = SimpleNamespace(close=AsyncMock())
        with patch.dict('os.environ', {'OPENAI_API_KEY':'test-key'}), patch('openai.AsyncOpenAI', return_value=fake_client):
            await session.finalize_input()
        assert session.domain._run_tool.await_args.args[1] == 'update_confirmed_booking'
        sent = session.domain._run_tool.await_args.args[2]
        assert sent['booking_id'] == 247
        assert sent['date'] == '2026-09-27'
        assert sent['caller_confirmed'] is True
        session.speak.assert_awaited_once()
    asyncio.run(run())


def test_failed_approved_action_does_not_loop_on_the_same_confirmation():
    async def run():
        session, _ = make_session()
        clear_call_memory(session.domain.session_id)
        begin_caller_turn(session.domain.session_id, 'Please read back the change')
        digest = register_pending_confirmation(session.domain.session_id, ACTION_UPDATE_CONFIRMED_BOOKING,
                                               {'booking_id':247, 'date':'2026-09-27'})
        assert release_pending_confirmation(session.domain.session_id, ACTION_UPDATE_CONFIRMED_BOOKING, digest)
        begin_caller_turn(session.domain.session_id, 'Yes, make that change')
        session.input_id = 'failed-action-turn'
        failed = ToolOutcome(name='update_confirmed_booking', call_id='failed-action-turn:update_confirmed_booking',
                             arguments={}, result=None, success=False, error='slot_unavailable')
        session.domain._run_tool.return_value = failed
        session.domain._with_confirmation = lambda outcome: outcome
        session.domain._sync_order_memory = AsyncMock(return_value=None)
        session.domain._model_tool_output = lambda outcome: {'status':'failed'}
        session.speak = AsyncMock()
        fake_client = SimpleNamespace(close=AsyncMock())
        with patch.dict('os.environ', {'OPENAI_API_KEY':'test-key'}), patch('openai.AsyncOpenAI', return_value=fake_client):
            await session.confirm_pending_action(ACTION_UPDATE_CONFIRMED_BOOKING,
                                                 get_pending_confirmation(session.domain.session_id, ACTION_UPDATE_CONFIRMED_BOOKING))
        assert get_pending_confirmation(session.domain.session_id, ACTION_UPDATE_CONFIRMED_BOOKING) is None
        assert session.speak.await_args.args[0].startswith('I could not verify the reservation change')
        assert session.domain._run_tool.await_count == 1
    asyncio.run(run())


def test_cancelled_call_never_reaches_domain():
    async def run():
        session, _ = make_session()
        session.cancelled_calls.add('cancelled')
        out = await session.execute_function({'id':'cancelled','name':'create_booking','args':{}},0)
        assert out['status'] == 'cancelled'
        session.domain._run_tool.assert_not_called()
    asyncio.run(run())


def test_interruption_during_transcript_wait_never_reaches_domain():
    async def run():
        session, _ = make_session()
        async def finalize(): await session.interrupt(provider=True)
        session.finalize_input = finalize
        out = await session.execute_function({'id':'old','name':'create_booking','args':{}},0)
        assert out['status'] == 'cancelled'
        session.domain._run_tool.assert_not_called()
    asyncio.run(run())


def test_name_disagreement_does_not_write():
    async def run():
        session, _ = make_session()
        session.finalize_input = AsyncMock()
        session.input_text = 'My name is Abubakar.'
        out = await session.execute_function({'id':'x','name':'update_reservation_draft','args':{'name':'Jacob'}},0)
        assert out['status'] == 'needs_clarification'
        session.domain._run_tool.assert_not_called()
    asyncio.run(run())


def test_plain_approval_cannot_rebuild_order():
    async def run():
        session, _ = make_session()
        session.finalize_input = AsyncMock()
        session.input_text = 'Yes, please confirm that.'
        session.apply_order_changes = AsyncMock()
        out = await session.execute_function({'id':'approval','name':'apply_order_changes','args':{'changes':[]}},0)
        assert out['error'] == 'confirmation_turn_cannot_change_order'
        session.apply_order_changes.assert_not_called()
        session.domain._run_tool.assert_not_called()
    asyncio.run(run())


def test_reservation_prepare_and_blocked_commit_return_authoritative_readback():
    async def run():
        for name in ('update_reservation_draft', 'create_booking'):
            session, _ = make_session()
            session.finalize_input = AsyncMock()
            session.input_text = 'My name is Abubakar. Please read back my requested reservation.'
            first = ToolOutcome(name=name, call_id='x', arguments={}, result={},
                success=name == 'update_reservation_draft', readback_verified=name == 'update_reservation_draft',
                error='confirmation_readback_not_released' if name == 'create_booking' else '')
            pending = ToolOutcome(name='get_reservation_draft', call_id='x:readback', arguments={}, result={},
                success=False, readback_verified=False, pending=True, confirmation_text='Would you like me to confirm the requested reservation for Abubakar?')
            session.domain._run_tool.side_effect = [first, pending]
            session.domain._with_confirmation = lambda outcome: outcome
            session.domain._sync_order_memory = AsyncMock(return_value=None)
            session.domain._model_tool_output = lambda outcome: {'status':'pending'}
            result = await session.execute_function({'id':'x','name':name,'args':{}},0)
            assert result['readback_text'] == pending.confirmation_text
            assert session.domain._run_tool.await_args_list[1].args[1] == 'get_reservation_draft'
            assert session.domain._outcomes == [first, pending]
            session.domain._release_pending_readbacks.assert_not_called()
    asyncio.run(run())


def test_service_error_code_is_actionable_without_exception_message():
    async def run():
        from app.services.restaurant import RestaurantServiceError
        executor = SimpleNamespace(invoke=AsyncMock(side_effect=RestaurantServiceError('Private exception details', code='slot_unavailable')))
        outcome = await ToolBridge(executor).invoke(call_id='full-slot', name='check_table_availability', arguments={'date':'2026-09-26','time':'19:30','party_size':4}, turn_id='turn', state_version=0)
        assert outcome.error == 'slot_unavailable'
        assert not outcome.success and not outcome.readback_verified
        assert 'Private exception details' not in str(outcome)
    asyncio.run(run())


def ready_readback(session, text='Would you like me to confirm this order?'):
    outcome = ToolOutcome(name='get_order_summary',call_id='summary',arguments={},result={'pending_confirmation_hash':'digest'},success=True,readback_verified=True,pending=True,confirmation_text=text,facts={'draft_version':4})
    session.utterance = {'token':'spoken','epoch':0,'text':text,'done':True,'played':False,'started':time.monotonic()-3,'bytes':48000,'version':0,'outcomes':(outcome,)}
    return outcome


def test_only_heard_matching_current_readback_releases_confirmation():
    async def run():
        session, _ = make_session()
        ready_readback(session)
        assert await session.played('spoken')
        session.domain._release_pending_readbacks.assert_awaited_once()
        assert session.heard_order['draft_version'] == 4
        assert not await session.played('spoken')
    asyncio.run(run())


def test_mismatched_spoken_readback_does_not_release():
    async def run():
        session, events = make_session()
        ready_readback(session)
        session.utterance['text'] = 'Everything is booked, Jacob.'
        assert await session.played('spoken')
        session.domain._release_pending_readbacks.assert_not_called()
        assert any(e.get('code') == 'spoken_readback_not_matched' for e in events)
    asyncio.run(run())


def reservation_outcome():
    proposed = {
        'customer_name':'Abubakar', 'customer_phone':'4155550198',
        'date':'2026-09-26', 'time':'19:30', 'party_size':4,
        'notes':'vegetarian seating note',
    }
    return ToolOutcome(
        name='get_reservation_draft', call_id='reservation-summary', arguments={},
        result={'proposed': proposed}, facts=proposed, success=True,
        readback_verified=True, pending=True,
        confirmation_text='Would you like me to confirm your reservation for 2026-09-26 at 19:30, for 4 guests, under Abubakar, using callback phone 4155550198, with notes vegetarian seating note?',
    )


def test_natural_complete_reservation_readback_is_eligible():
    spoken = ('I have Saturday at seven thirty PM for four guests under Abubakar, '
              'callback number 415-555-0198, with your vegetarian seating note. '
              'Would you like me to confirm and book that?')
    assert reservation_readback_matches(reservation_outcome(), spoken)


def test_reservation_readback_missing_identity_or_claiming_success_is_blocked():
    outcome = reservation_outcome()
    assert not reservation_readback_matches(
        outcome, 'Saturday at seven thirty for four guests. Would you like me to confirm?'
    )
    assert not reservation_readback_matches(
        outcome, 'Your reservation has been confirmed for Saturday at seven thirty under Abubakar, phone 4155550198, for four guests, vegetarian note.'
    )


def test_natural_complete_reservation_readback_releases_confirmation():
    async def run():
        session, _ = make_session()
        outcome = reservation_outcome()
        session.utterance = {
            'token':'reservation-spoken', 'epoch':0,
            'text':('Saturday at seven thirty PM, four guests under Abubakar, '
                    'callback 415 555 0198, vegetarian seating note. '
                    'Would you like me to confirm and book it?'),
            'done':True, 'played':False, 'started':time.monotonic()-3,
            'bytes':48000, 'version':0, 'outcomes':(outcome,),
        }
        assert await session.played('reservation-spoken')
        session.domain._release_pending_readbacks.assert_awaited_once_with(
            outcome.confirmation_text, 0
        )
    asyncio.run(run())


def test_early_and_interrupted_playback_never_release():
    async def run():
        session, _ = make_session()
        ready_readback(session)
        session.utterance['started'] = time.monotonic()
        assert not await session.played('spoken')
        await session.interrupt(provider=True)
        assert not await session.played('spoken')
        assert session.block_output
        session.domain._release_pending_readbacks.assert_not_called()
    asyncio.run(run())


def test_native_caption_is_not_menu_prompt_or_separate_recognizer():
    async def run():
        session, events = make_session()
        await session.input_caption('My name is ')
        await session.input_caption('Abubakar.')
        assert session.input_text == 'My name is Abubakar.'
        assert [e['delta'] for e in events if e['type']=='transcript_delta'] == ['My name is ','Abubakar.']
        session.socket.send.assert_not_called()
    asyncio.run(run())


@pytest.mark.skipif(os.getenv('RUN_DB_INTEGRATION') != '1', reason='marked disposable PostgreSQL required')
@pytest.mark.parametrize('spoken, expected_status, expected_action', [
    ('Ya ya.', 'confirmed', 'confirm_order'),
    ('Yeah, leave that pre-order. I will order on arrival. Just book my reservation.',
     'cancelled', 'abandon_pending_order'),
])
@pytest.mark.asyncio
async def test_real_call_order_decision_is_scoped_and_committed_once(spoken, expected_status, expected_action):
    from datetime import datetime
    from app.native_voice.database_guard import close_native_voice_pool, get_native_voice_pool
    from app.native_voice.protocol import MemoryRealtimeTransport
    from app.services.restaurant import RestaurantService

    call_id = 'gemini-action-eval-' + uuid.uuid4().hex
    pool = await get_native_voice_pool()
    adapter = None
    booking_id = None
    try:
        async with pool.acquire() as conn:
            table_id = await conn.fetchval('SELECT id FROM tables ORDER BY id LIMIT 1')
            booking_id = await conn.fetchval(
                'INSERT INTO bookings (customer_name, customer_phone, table_id, booked_at, party_size, status) '
                'VALUES ($1, $2, $3, $4, $5, $6) RETURNING id',
                'Synthetic Gemini Guest', '+15035550198', table_id,
                datetime(2026, 9, 26, 18, 30), 2, 'confirmed',
            )
            await conn.execute('INSERT INTO call_sessions (session_id, state) VALUES ($1, $2::jsonb)',
                               call_id, json.dumps({
                                   'booking_id': booking_id,
                                   'customer_name': 'Synthetic Gemini Guest',
                                   'customer_phone': '+15035550198',
                                   'reservation_draft': {
                                   'booking_id': booking_id, 'status': 'confirmed',
                                   'customer_name': 'Synthetic Gemini Guest',
                                   'customer_phone': '+15035550198',
                                   'date': '2026-09-26', 'time': '18:30', 'party_size': 2,
                               }}))
        service = RestaurantService(pool_provider=get_native_voice_pool)
        await service.add_order_item(call_id=call_id, idempotency_key=call_id + '-salad',
                                     item_name='Chicken Caesar Salad', quantity=2)
        await service.add_order_item(call_id=call_id, idempotency_key=call_id + '-noodles',
                                     item_name='Chilled Peanut Noodles', quantity=1)
        await service.set_order_fulfillment(call_id=call_id, idempotency_key=call_id + '-dine-in',
                                            fulfillment_type='dine_in', booking_id=booking_id)
        summary = await service.get_order_summary(call_id=call_id)
        assert int(summary['booking_id']) == booking_id
        assert summary['status'] == 'pending'
        adapter = NativeVoiceAdapter(session_id=call_id, transport=MemoryRealtimeTransport())
        await adapter.start()
        record = get_pending_confirmation(call_id, ACTION_CONFIRM_ORDER)
        assert record and release_pending_confirmation(call_id, ACTION_CONFIRM_ORDER, record['payload_hash'])
        await adapter._persist_native_confirmation_state()
        events = []
        async def emit(event): events.append(event)
        socket = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
        session = GeminiLiveSession(domain=adapter, socket=socket, send=emit, pace='natural')
        session.input_id = 'gemini-approval-turn'
        session.input_text = spoken
        session.input_ready.set()
        session.input_updated = time.monotonic() - 1
        session.speak = AsyncMock()
        fake_client = SimpleNamespace(close=AsyncMock())
        with patch.dict('os.environ', {'OPENAI_API_KEY':'test-key'}), patch('openai.AsyncOpenAI', return_value=fake_client):
            await session.finalize_input()
        async with pool.acquire() as conn:
            order = await conn.fetchrow('SELECT id, status, booking_id, total_amount FROM orders WHERE session_id = $1', call_id)
            items = await conn.fetch('SELECT item_name, quantity FROM order_items WHERE order_id = $1 ORDER BY item_name', order['id'])
            actions = await conn.fetch('SELECT action FROM voice_action_idempotency WHERE call_id = $1 AND action IN ($2, $3)',
                                       call_id, 'confirm_order', 'abandon_pending_order')
        assert order['status'] == expected_status, [event.get('error') for event in events if event.get('type') == 'tool']
        assert order['booking_id'] == booking_id
        assert (float(order['total_amount']) > 0) == (expected_status == 'confirmed')
        assert [(row['item_name'], row['quantity']) for row in items] == [('Chicken Caesar Salad', 2), ('Chilled Peanut Noodles', 1)]
        assert [row['action'] for row in actions] == [expected_action]
        assert any(event.get('type') == 'tool' and event.get('name') == expected_action and event.get('verified') for event in events)
        assert adapter.state.status == expected_status
        session.speak.assert_awaited_once()
    finally:
        if adapter is not None:
            await adapter.close()
        async with pool.acquire() as conn:
            await conn.execute('DELETE FROM voice_action_idempotency WHERE call_id = $1', call_id)
            await conn.execute('DELETE FROM orders WHERE session_id = $1', call_id)
            await conn.execute('DELETE FROM call_sessions WHERE session_id = $1', call_id)
            if booking_id is not None:
                await conn.execute('DELETE FROM bookings WHERE id = $1', booking_id)
        clear_call_memory(call_id)
        await close_native_voice_pool()


@pytest.mark.skipif(os.getenv('RUN_DB_INTEGRATION') != '1', reason='marked disposable PostgreSQL required')
@pytest.mark.asyncio
async def test_real_call_sunday_approval_updates_same_booking_without_repeat_prompt():
    from datetime import datetime
    from app.native_voice.database_guard import close_native_voice_pool, get_native_voice_pool
    from app.native_voice.protocol import MemoryRealtimeTransport
    from app.services.restaurant import RestaurantService

    call_id = 'gemini-update-eval-' + uuid.uuid4().hex
    pool = await get_native_voice_pool()
    adapter = None
    booking_id = None
    try:
        async with pool.acquire() as conn:
            table_id = await conn.fetchval('SELECT id FROM tables ORDER BY id LIMIT 1')
            booking_id = await conn.fetchval(
                'INSERT INTO bookings (customer_name, customer_phone, table_id, booked_at, party_size, status) '
                'VALUES ($1, $2, $3, $4, $5, $6) RETURNING id',
                'Synthetic Sunday Guest', '+15035550197', table_id,
                datetime(2026, 9, 26, 18, 30), 2, 'confirmed',
            )
            await conn.execute('INSERT INTO call_sessions (session_id, state) VALUES ($1, $2::jsonb)',
                               call_id, json.dumps({
                                   'booking_id': booking_id,
                                   'customer_name': 'Synthetic Sunday Guest',
                                   'customer_phone': '+15035550197',
                                   'reservation_draft': {
                                       'booking_id': booking_id, 'status': 'confirmed',
                                       'customer_name': 'Synthetic Sunday Guest',
                                       'customer_phone': '+15035550197',
                                       'date': '2026-09-26', 'time': '18:30', 'party_size': 2,
                                   },
                               }))
        service = RestaurantService(pool_provider=get_native_voice_pool)
        proposed = await service.update_confirmed_booking(
            call_id=call_id, idempotency_key=call_id + '-proposal',
            booking_id=booking_id, confirmed=False, date='2026-09-27')
        assert proposed['pending']
        adapter = NativeVoiceAdapter(session_id=call_id, transport=MemoryRealtimeTransport())
        await adapter.start()
        record = get_pending_confirmation(call_id, ACTION_UPDATE_CONFIRMED_BOOKING)
        assert record and release_pending_confirmation(call_id, ACTION_UPDATE_CONFIRMED_BOOKING, record['payload_hash'])
        await adapter._persist_native_confirmation_state()
        events = []
        async def emit(event): events.append(event)
        socket = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
        session = GeminiLiveSession(domain=adapter, socket=socket, send=emit, pace='natural')
        session.input_id = 'gemini-sunday-approval-turn'
        session.input_text = 'Yes, make that change.'
        session.input_ready.set()
        session.input_updated = time.monotonic() - 1
        session.speak = AsyncMock()
        fake_client = SimpleNamespace(close=AsyncMock())
        with patch.dict('os.environ', {'OPENAI_API_KEY':'test-key'}), patch('openai.AsyncOpenAI', return_value=fake_client):
            await session.finalize_input()
        async with pool.acquire() as conn:
            booked = await conn.fetchrow('SELECT id, booked_at, party_size, status FROM bookings WHERE id = $1', booking_id)
            action_count = await conn.fetchval('SELECT count(*) FROM voice_action_idempotency WHERE call_id = $1 AND action = $2', call_id, 'update_confirmed_booking')
        assert booked['booked_at'] == datetime(2026, 9, 27, 18, 30), [event.get('error') for event in events if event.get('type') == 'tool']
        assert booked['id'] == booking_id and booked['status'] == 'confirmed'
        assert action_count == 1
        assert any(event.get('type') == 'tool' and event.get('name') == 'update_confirmed_booking' and event.get('verified') for event in events)
        session.speak.assert_awaited_once()
    finally:
        if adapter is not None:
            await adapter.close()
        async with pool.acquire() as conn:
            await conn.execute('DELETE FROM voice_action_idempotency WHERE call_id = $1', call_id)
            await conn.execute('DELETE FROM call_sessions WHERE session_id = $1', call_id)
            if booking_id is not None:
                await conn.execute('DELETE FROM bookings WHERE id = $1', booking_id)
        clear_call_memory(call_id)
        await close_native_voice_pool()
