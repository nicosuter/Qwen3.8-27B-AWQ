# Scoring LiveCodeBench away from the model

Scoring this suite means running code the model wrote. The tests are the
verifier, so execution is not optional -- it can only be put somewhere it cannot
do damage. That is not the cluster serving the model, which holds the
checkpoints, the frozen item sets and every result produced so far.

`eval/scripts/adapters/livecodebench.py run --defer-execution` generates and stops.
This runs the other half.

## The split

    GPU cluster                          here
    -----------                          ----
    run --defer-execution                score
      writes generations/<suite>-           reads generations + answer key
        <variant>-r<n>.jsonl                executes, writes results + metadata
      no code executes                      no network at all

Two files cross the boundary, plus the answer key. The scorer needs nothing
else, which is why it can be denied all networking rather than given an
allow-list that someone widens later.

## Containment

- Its own namespace, `restricted` pod security, and deliberately outside
  ArgoCD, so nothing reconciles it away and it cannot reach what ArgoCD manages.
- `NetworkPolicy` denying ingress and egress with no exceptions, DNS included.
- Non-root, all capabilities dropped, no privilege escalation, read-only root
  filesystem, `RuntimeDefault` seccomp.
- `automountServiceAccountToken: false`. A solution that goes looking for the
  Kubernetes API finds neither a token nor a route to one.
- Generated code writes to an `emptyDir` at `/scratch`, not to the volume
  holding the inputs.
- CPU, memory and ephemeral storage are capped; the adapter additionally applies
  its own per-process CPU-time and address-space limits and a wall-clock timeout.

None of this is a kernel boundary. The cluster has no gVisor or Kata runtime
class, so a container escape would land on the node. The threat being managed is
a generated program that loops, forks, fills a disk or reaches for something on
the network, not a targeted exploit.

## Running it

    kubectl apply -f eval/k8s/livecodebench-score/

    # refresh the adapter after any change to it
    kubectl create configmap lcb-adapter -n qwen-lcb-score \
      --from-file=livecodebench.py=eval/scripts/adapters/livecodebench.py \
      --from-file=_common.py=eval/scripts/adapters/_common.py \
      --dry-run=client -o yaml | kubectl apply -f -

Put `generations.jsonl`, its `.meta.json` and the suite's `.key.json` on the
`lcb-work` volume, then run the job. Results land beside them as
`results.jsonl` and `metadata.json`, ready to be copied back into the run
directory the generation came from.

The adapter is stdlib-only, so the image needs nothing installed and the pod
never fetches anything.
