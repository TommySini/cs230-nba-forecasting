# Draft response for AWS Support case 179411216100407

Draft only; not submitted. Reopen the existing case in the Support Center.

Operator note after the new three-region email: confirm the account's permitted
EC2 regions before choosing where to run. The latest pasted denial has no case
number; it must not be assumed to refer to N. Virginia case 179411421300610.

Hello,

Please reconsider our request to increase the Oregon (us-west-2) Running
On-Demand G and VT instance quota from 0 to 8 vCPUs.

We are Tommaso Sini and Anthony Viverito, working on a Stanford CS230 course
project on NBA game forecasting. Our repository is
https://github.com/TommySini/cs230-nba-forecasting.

We need one g6e.2xlarge instance, which requires 8 G-family vCPUs and provides
one 48 GB L40S GPU. The workload is inference with Qwen2.5-14B-Instruct followed
by 4-bit, rank-8 LoRA adapter training and a small probability head. The first
learning diagnostic uses only 16–32 training games. Research and training will
run sequentially on one GPU worker, without autoscaling or concurrent GPU workers.

We understand your concern about unexpected bills. The entire experiment has
a USD 200 cap including previously tracked costs, while the AWS account
currently has only USD 100 of verified earned account-plan credits. We will
not treat potential future credits as available funds or launch workloads
until applicable credit coverage is confirmed. We are implementing serialized
cost reservations, maximum request-token budgets, explicit runtime ceilings,
and external instance termination before any paid launch. The initial joint
training diagnostic is capped at 30 minutes, with separate admission required
for setup, research and other infrastructure costs.

Project-specific EC2 instance profiles are already deployed with restricted
TRAIN artifact access. No GPU instance or paid inference request has been
launched. This is a small academic experiment, with no production endpoint,
cryptocurrency mining, or automated wagering workload.

The requested quota would allow exactly one planned worker. If 8 vCPUs cannot
be approved, would an initial **4 G-family vCPUs** be possible for one
g6e.xlarge? It has the same single 48 GB L40S GPU with less system memory; we
would validate that smaller configuration before running the training pilot.
Please also confirm which three EC2 regions are currently permitted for this
account. N. Virginia, Ohio and Oregon would suit us; we only need one region
for execution, not concurrent regional deployments. If another account
verification step is required, please identify it.

Thank you for reviewing the additional details.
