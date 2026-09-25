import asyncio
from contextlib import suppress
from dataclasses import replace
from unittest.mock import AsyncMock,patch
import pytest
from test_native_voice_gemini import make_session
from app.native_voice.quantity import party_count,item_count
from app.native_voice.contracts import OrderState,OrderItemState
from app.native_voice.tools import ToolOutcome
from app.pending_confirmation import classify_affirmation

@pytest.mark.parametrize('text',["Yes, is parking free?","Okay, what is the total?","Yes if you have a window table","Sure, I need another minute","Yes, and two more soups","Okay, a table for four","Yes, but make it five"])
def test_non_approval_cannot_authorize(text):
 assert classify_affirmation(text)!='affirmative'

@pytest.mark.parametrize('text,count',[('a couple of Tomato Soups',2),('two Tomato Soups',2),('2 Tomato Soups',2),('half a dozen Tomato Soups',6),('a pair of Tomato Soups',2),('triple Tomato Soup',3)])
def test_number_prefixes(text,count):
 assert item_count(text,'Tomato Soup')==count

def test_relative_counts_and_noncounts():
 assert item_count('two more Tomato Soups','Tomato Soup',3)==5
 assert item_count('the second Tomato Soup','Tomato Soup',3) is None
 assert item_count('my phone is 5035550188, pickup at 7:30','Tomato Soup') is None
 assert party_count('a table for a couple')==2
 assert party_count('add two more guests',3)==5
 assert party_count('two fewer guests',5)==3

def test_controller_overrides_wrong_relative_line_quantity():
 async def run():
  s,_=make_session();s.finalize_input=AsyncMock();s.reservation_identity=AsyncMock(return_value={})
  s.input_text='Please make that two more Tomato Soups.'
  s.domain.state=OrderState(items=(OrderItemState(canonical_item_id='soup',item_name='Tomato Soup',quantity=3,line_id='12'),))
  s.apply_order_changes=AsyncMock(return_value=({'status':'completed'},None))
  await s.execute_function({'id':'qty','name':'apply_order_changes','args':{'actions':[{'name':'update_order_item','arguments':{'order_item_id':12,'quantity':2}}]}},0)
  assert s.apply_order_changes.await_args.args[0]['actions'][0]['arguments']['quantity']==5
 asyncio.run(run())

def test_routine_edit_does_not_request_full_readback():
 async def run():
  s,_=make_session();s.finalize_input=AsyncMock();s.reservation_identity=AsyncMock(return_value={});s.input_text='Change the Tomato Soup to two.'
  o=ToolOutcome(name='get_order_summary',call_id='summary',arguments={},result={},success=True,pending=True,confirmation_text='FULL ORDER READBACK')
  s.apply_order_changes=AsyncMock(return_value=({'speech':'FULL ORDER READBACK','status':'pending_confirmation'},o))
  result=await s.execute_function({'id':'edit','name':'apply_order_changes','args':{'actions':[]}},0)
  assert 'readback_text' not in result and 'speech' not in result
  assert 'Do not recite' in result['instruction']
 asyncio.run(run())

def test_checkout_still_requires_complete_readback():
 async def run():
  s,_=make_session();s.finalize_input=AsyncMock();s.reservation_identity=AsyncMock(return_value={});s.input_text="That's all, review my order."
  o=ToolOutcome(name='get_order_summary',call_id='summary',arguments={},result={},success=True,pending=True,confirmation_text='FULL ORDER READBACK')
  s.apply_order_changes=AsyncMock(return_value=({'status':'pending_confirmation'},o))
  result=await s.execute_function({'id':'edit','name':'apply_order_changes','args':{'actions':[]}},0)
  assert result['readback_text']=='FULL ORDER READBACK'
 asyncio.run(run())

def test_amendment_prompt_spoken_by_server_not_model():
 async def run():
  s,_=make_session();s.pending_booking_update={'party_size':5};done=asyncio.Event()
  s.execute_function=AsyncMock(return_value={'readback_text':'Change to five guests. Should I make that change?','reservation_amendment_prompt':True})
  async def speak(*args,**kw):done.set()
  s.speak=AsyncMock(side_effect=speak)
  class Client:
   def __init__(self,**kw):pass
   async def __aenter__(self):return self
   async def __aexit__(self,*args):pass
  with patch('openai.AsyncOpenAI',Client),patch.dict('os.environ',{'OPENAI_API_KEY':'test'}):
   worker=asyncio.create_task(s.tools_worker())
   await s.tool_queue.put(([{'id':'change','name':'update_confirmed_booking'}],0))
   await asyncio.wait_for(done.wait(),2)
   assert s.block_output and s.speak.await_args.args[0].startswith('Change to five')
   worker.cancel()
   with suppress(asyncio.CancelledError):await worker
 asyncio.run(run())


def test_conflicting_counts_are_not_overridden_by_first_match():
 assert item_count('two Tomato Soups, actually three Tomato Soups','Tomato Soup') is None

def test_paid_change_readback_is_not_suppressed():
 async def run():
  s,_=make_session();s.finalize_input=AsyncMock();s.reservation_identity=AsyncMock(return_value={});s.input_text='Change the soup to two.'
  o=ToolOutcome(name='update_order_item',call_id='paid',arguments={},result={},success=True,pending=True,confirmation_text='Approve the paid change?')
  s.apply_order_changes=AsyncMock(return_value=({'status':'pending_confirmation'},o))
  result=await s.execute_function({'id':'paid-edit','name':'apply_order_changes','args':{'actions':[]}},0)
  assert result['readback_text']=='Approve the paid change?'
 asyncio.run(run())

def test_explicit_identity_and_multilingual_phone_prefix_are_retained():
 from app.native_voice.gemini_live import extract_identity_slots
 assert extract_identity_slots('I would like a table under the name Alex Khan. For Saturday.') == {'name':'Alex Khan'}
 assert extract_identity_slots('या 3335552154',reservation_context=True) == {'phone':'3335552154'}
 assert extract_identity_slots('Yaar 3',reservation_context=True) == {}
 assert extract_identity_slots('My colleague is Alex Khan') == {}

def test_prior_identity_is_usable_without_asking_again():
 async def run():
  s,_=make_session()
  s.reservation_buffer={'name':'Alex Khan','phone':'3335552154'}
  s.input_history=['A table under the name Alex Khan.','या 3335552154']
  s.input_text='Can you book my reservation?'
  assert await s.validate_identity_fields({'name':'Alex Khan','phone':'+13335552154'},confirmation_only=False) is None
 asyncio.run(run())

def test_exact_transcript_name_correction_updates_draft():
 async def run():
  s,_=make_session();s.reservation_identity=AsyncMock(return_value={'customer_name':'Abubakar','customer_phone':'+13335552154'})
  s.input_text='¿Qué? No, under the name Alex Khan.'
  assert await s.validate_identity_fields({'name':'Alex Khan','phone':'3335552154'},confirmation_only=False) is None
  assert s.reservation_buffer['name']=='Alex Khan'
 asyncio.run(run())

def test_unrelated_historical_name_is_not_accepted_as_reservation_identity():
 async def run():
  s,_=make_session();s.input_history=['My colleague is Alex Khan.'];s.input_text='Book my reservation.'
  result=await s.validate_identity_fields({'name':'Alex Khan'},confirmation_only=False)
  assert result['error']=='name_not_in_current_speech'
 asyncio.run(run())

@pytest.mark.parametrize('spoken,expected',[
 ('triple five','555'),('double five','55'),('treble three','333'),
 ('3335555432one','33355554321'),('triple three double five four three two one','333554321')])
def test_phone_digit_repeats_and_adjoining_words(spoken,expected):
 from app.native_voice.gemini_live import phone_digits
 assert phone_digits(spoken)==expected

def test_truncated_phone_cannot_pass_substring_validation():
 async def run():
  s,_=make_session();s.input_text='3335555432one'
  result=await s.validate_identity_fields({'phone':'3335555432'},confirmation_only=False)
  assert result['error']=='phone_transcript_mismatch'
 asyncio.run(run())

def test_unclear_repeat_word_cannot_authorize_phone_guess():
 async def run():
  s,_=make_session();s.input_text='333 tetra five 4321'
  result=await s.validate_identity_fields({'phone':'3335543210'},confirmation_only=False)
  assert result['error']=='ambiguous_digit_repeat'
 asyncio.run(run())

def test_menu_time_uses_reservation_but_not_for_separate_pickup():
 async def run():
  s,_=make_session();s.reservation_buffer={'date':'2026-09-27','time':'19:30'};s.input_text='Which salads can I pre-order?'
  assert (await s.menu_moment()).isoformat()=='2026-09-27T19:30:00-07:00'
  s.domain.state=replace(s.domain.state,fulfillment='pickup',fulfillment_details={})
  assert await s.menu_moment() is None
  s.domain.state=replace(s.domain.state,fulfillment_details={'fulfillment_at':'2026-09-25T18:00:00-07:00'})
  assert (await s.menu_moment()).isoformat()=='2026-09-25T18:00:00-07:00'
 asyncio.run(run())

@pytest.mark.parametrize('text',["a table for a four people", "a table for four people and one order of noodles", "a table for a 4 people"])
def test_filler_article_does_not_replace_guest_count(text):
 assert party_count(text)==4

@pytest.mark.parametrize('text',["Yes, but you forget my pre-order.","You forgot the order.","Don't forget my pre-order.","Do not cancel the order."])
def test_missing_order_complaint_is_not_abandonment(text):
 from app.pending_confirmation import requests_order_abandonment
 assert not requests_order_abandonment(text)

@pytest.mark.parametrize('text',["Cancel my pre-order.","Please skip the food order.","Forget the order."])
def test_explicit_order_abandonment_remains_supported(text):
 from app.pending_confirmation import requests_order_abandonment
 assert requests_order_abandonment(text)

def test_goodbye_is_scoped_and_ending_session_ignores_noise():
 from app.native_voice.gemini_live import is_call_farewell
 assert is_call_farewell('No, thank you. Bye.')
 assert not is_call_farewell('Before I say goodbye, change my reservation.')
 async def run():
  s,events=make_session();s.ending_call=True
  await s.input_caption('¿Qué?');await s.append_audio(b'\x00\x00')
  s.socket.send.assert_not_called()
  assert not events
 asyncio.run(run())

def test_explicit_goodbye_closes_input_before_speaking():
 from types import SimpleNamespace
 from unittest.mock import Mock
 async def run():
  s,events=make_session();s.input_text='No, thank you. Bye.';s.input_id='bye';s.input_ready.set()
  s.domain.turns=SimpleNamespace(start=Mock())
  s.domain._finalize_caller_turn=AsyncMock();s.speak=AsyncMock()
  class Client:
   def __init__(self,**kw):pass
   async def __aenter__(self):return self
   async def __aexit__(self,*args):pass
  with patch('openai.AsyncOpenAI',Client),patch.dict('os.environ',{'OPENAI_API_KEY':'test'}):
   await s.finalize_input()
  assert s.ending_call and s.block_output
  assert [e['type'] for e in events][-2:]==['call_closing','call_end_after_playback']
  s.speak.assert_awaited_once()
  old=s.input_text
  await s.input_caption('¿Qué?')
  assert s.input_text==old
 asyncio.run(run())

@pytest.mark.parametrize('text',["Yep. Yes.","Yes, yes.","Okay, yep.","Yeah, you can finalize my booking.","Please confirm my reservation.","Go ahead and confirm the reservation."])
def test_natural_single_intent_approvals(text):
 assert classify_affirmation(text)=='affirmative'

@pytest.mark.parametrize('text',["Yep. Yes, but make it five.","Yes, yes, is parking free?","Can you confirm my reservation?","Please confirm my reservation if the food is available.","Yeah, you can finalize my booking, but change the name."])
def test_mixed_or_conditional_approvals_stay_blocked(text):
 assert classify_affirmation(text)!='affirmative'

def test_first_availability_query_uses_caller_slots_not_model_date():
 async def run():
  s,_=make_session();s.finalize_input=AsyncMock()
  s.current_reservation_slots={'date':'2026-09-27','time':'19:30','party_size':4}
  s.input_text='A table for four this Sunday at 7:30 p.m.'
  outcome=ToolOutcome(name='check_table_availability',call_id='date',arguments={},result={'available':True},success=True,readback_verified=True)
  s.domain._run_tool=AsyncMock(return_value=outcome)
  s.domain._with_confirmation=lambda o:o
  s.domain._persist_committed_outcome=AsyncMock(return_value=outcome)
  s.domain._sync_order_memory=AsyncMock(return_value=None)
  s.domain._model_tool_output=lambda o:{'status':'completed'}
  await s.execute_function({'id':'date','name':'check_table_availability','args':{'date':'2026-09-28','time':'19:00','party_size':1}},0)
  sent=s.domain._run_tool.await_args.args[2]
  assert (sent['date'],sent['time'],sent['party_size'])==('2026-09-27','19:30',4)
 asyncio.run(run())

def test_ambiguous_approval_is_not_reported_as_system_failure():
 async def run():
  s,_=make_session();s.finalize_input=AsyncMock();s.input_text='Yes, is parking free?'
  result=await s.execute_function({'id':'ambiguous','name':'create_booking','args':{'caller_confirmed':True}},0)
  assert result['status']=='needs_confirmation'
  assert 'No technical failure' in result['instruction']
  s.domain._run_tool.assert_not_called()
 asyncio.run(run())

def test_order_readback_is_server_owned_without_model_tool_choice():
 from types import SimpleNamespace
 from unittest.mock import Mock
 async def run():
  s,_=make_session();s.input_text='Please read my food order back for approval.';s.input_id='review';s.input_ready.set()
  s.domain.state=OrderState(items=(OrderItemState(canonical_item_id='salad',item_name='Salad',quantity=2,line_id='1'),))
  s.domain.turns=SimpleNamespace(start=Mock());s.domain._finalize_caller_turn=AsyncMock()
  s.execute_function=AsyncMock(return_value={'readback_text':'Two salads. Confirm?'})
  s.speak=AsyncMock()
  class Client:
   def __init__(self,**kw):pass
   async def __aenter__(self):return self
   async def __aexit__(self,*args):pass
  with patch('openai.AsyncOpenAI',Client),patch.dict('os.environ',{'OPENAI_API_KEY':'test'}):
   await s.finalize_input()
  assert s.execute_function.await_args.args[0]['name']=='get_order_summary'
  assert s.speak.await_args.args[0]=='Two salads. Confirm?'
  assert s.block_output
 asyncio.run(run())

def test_order_of_the_prefix_retains_quantity():
 assert item_count('one order of the Chilled Peanut Noodles','Chilled Peanut Noodles')==1
 assert item_count('two orders of Chicken Caesar Salad','Chicken Caesar Salad')==2

def test_provider_interrupt_cannot_strand_an_already_received_new_turn():
 async def run():
  s,_=make_session();s.input_text='Okay, thank you. Bye.';s.input_finalized=False
  await s.interrupt(provider=True)
  assert not s.block_output
  s.input_finalized=True
  await s.interrupt(provider=True)
  assert s.block_output
 asyncio.run(run())
