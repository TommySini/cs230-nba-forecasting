# Ready-to-send AWS support messages

Drafts only; not submitted. User prioritizes getting moving and delegates
region/instance selection. Prefer N. Virginia (us-east-1), with Ohio (us-east-2)
and Oregon (us-west-2) as the three requested permitted EC2 regions. No region
swap or new quota request has been submitted by these drafts.

## 1. Existing N. Virginia quota case

Open case 179411421300610 in the AWS Support Center. Add correspondence; if
closed, reopen if the console permits. Use the Support Center, not the
no-reply email address.

Hello,

Please reconsider this request with a smaller initial allocation of **4 vCPUs
under Running On-Demand G and VT instances in us-east-1**, sufficient for one
g6e.xlarge (one 48 GB L40S GPU). Our previous request was for 8 vCPUs.

We are Tommaso Sini and Anthony Viverito, working on a Stanford CS230 academic
NBA forecasting project: https://github.com/TommySini/cs230-nba-forecasting.
We plan a short pilot on 16–32 training games using Qwen2.5-14B with 4-bit LoRA
and a small probability head. We would validate the smaller instance's memory
and runtime first, then run one GPU worker at a time. There is no production
endpoint or autoscaling. The total project budget is USD 200, with only USD 100
of earned AWS credits currently verified. We are implementing bounded job
budgets and automatic termination before launching compute.

Your welcome email says EC2 launches are temporarily limited to three regions.
Please confirm or enable **N. Virginia, Ohio and Oregon** as our permitted
regions. We need to run in only one region and are flexible if approval is
possible sooner in another of those regions.

If this minimal GPU allocation cannot be approved, please identify the specific
account-verification step or smallest suitable allocation needed to proceed.

Thank you.

## 2. Account-access inquiry for Bedrock

Use an existing relevant account-access case if present; otherwise create an
Account and billing support case. Do not purchase a technical-support plan for
this account-authorization inquiry.

Subject: Bedrock account authorization remains blocked after payment fix

Hello,

I updated my payment method, and the Claude model agreements now show AVAILABLE;
the earlier INVALID_PAYMENT_INSTRUMENT error has cleared. However, Bedrock
still reports authorizationStatus=NOT_AUTHORIZED, including for AWS-sold
gpt-oss-120b and DeepSeek V3.2 in us-east-1. One minimal gpt-oss-120b Converse
request returned ValidationException: Operation not allowed.

Please identify and resolve any remaining account-verification or Bedrock
authorization requirement, or route this case to the responsible team. This
blocks a small Stanford CS230 academic project. Account ID: <aws-account-id>.

Thank you.

## After submission

Renew the expired existing cs230 sign-in before the agent performs more live
account work. The toolkit installation and Codex-only profile binding must be
preserved. CPU-only TRAIN preparation can proceed after actual CPU quota,
permitted region, scoped execution and bounded costs are verified; it does not
depend on GPU approval or Bedrock. No current quota was obtained in the latest
turn because authentication had expired. The smaller GPU instance is not yet
qualified and the existing production gate must not be bypassed to launch it.
