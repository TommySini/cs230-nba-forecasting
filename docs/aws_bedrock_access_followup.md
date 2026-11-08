# Draft Bedrock authorization inquiry

Draft only; not submitted. Use if authorization remains unavailable after the
recent payment update and model-activation processing. Do not buy paid support
or send account credentials. The observed state does not establish its internal
AWS cause.

Subject: Bedrock authorization remains NOT_AUTHORIZED after model agreements are available

Hello,

We are running a small Stanford CS230 academic NBA forecasting project in AWS
account <aws-account-id>. We need bounded serverless inference to prepare and review
training explanations, with no production endpoint or automated wagering.

The account plan API reports PAID / ACTIVE. On 2026-11-08 we corrected the
payment method and retried the already-selected Sonnet 4.6 and Opus 4.6 public
model agreements. The earlier INVALID_PAYMENT_INSTRUMENT error cleared. At
05:07 UTC, both models in us-west-2 returned:

- agreementAvailability.status: AVAILABLE
- entitlementAvailability: AVAILABLE
- regionAvailability: AVAILABLE
- authorizationStatus: NOT_AUTHORIZED

AWS-sold models checked in us-east-1 (DeepSeek V3.2, OpenAI gpt-oss-120b and
Amazon Nova Pro) also return AVAILABLE for agreement, entitlement and region,
but NOT_AUTHORIZED for authorization. We are happy to use these alternatives.
The Anthropic first-use form has already been submitted with the true use case.
After those checks, one minimal `bedrock-runtime:Converse` request in us-east-1
to `openai.gpt-oss-120b-1:0` (text: "Reply OK.", maxTokens:16, standard tier)
was rejected with:

`ValidationException: An error occurred (ValidationException) when calling the Converse operation: Operation not allowed`

There was no successful inference or returned usage. Please distinguish this
error from the earlier Marketplace payment error, which has cleared.

Please identify whether additional account-level Bedrock access or verification
is required, and the specific action needed to authorize model use. AWS's model
access documentation notes that model activation and additional account-level
access can be separate. Please route this inquiry to the team able to resolve
that account-level access if applicable.

Our total project cap is USD 200 including prior tracked work, with USD 100 of
confirmed earned AWS credit. We will not treat potential future credits as
available funds or launch above the verified funding limit.

Thank you.
