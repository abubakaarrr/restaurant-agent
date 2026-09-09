# Restaurant Knowledge Fallback

The canonical source for restaurant facts is the versioned synthetic fixture
`db/fixtures/harbor_and_hearth.v1.json`. Runtime restaurant-information tools
query its topic records; this summary exists only for the optional offline
embedding rebuild.

## Knowledge safety

Use current effective topic records for hours, reservations, parking,
accessibility, fulfillment, events, payments, and policies. Do not infer an
answer from an unrelated section. Unknown or stale information stays unknown;
transfer is offered only when the server has a configured destination.
