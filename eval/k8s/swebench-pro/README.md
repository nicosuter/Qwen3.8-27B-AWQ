# Running SWE-bench Pro rollouts on Kubernetes

A coding agent, driven by the model under test, with a shell in a container
built from someone else's image. The LiveCodeBench scorer next door contains one
untrusted thing; this contains two.

## Why here and not on the GPU cluster

Harbor's Singularity backend runs an HTTP server *inside* the task container and
health-checks `http://localhost:<port>/health` from the host, so the container
has to share the host network namespace. `--net --network none` severs the
control channel and every task fails. Denying the network there would mean
patching Harbor's transport onto a bind-mounted UNIX socket.

The Kubernetes backend has no such problem: it drives task containers through
`connect_get_namespaced_pod_exec`. Nothing listens inside the container, so the
task pod can be denied the network completely and still be driven.

## Two pieces of configuration that do nothing

**`network_mode` and `allow_internet`.** Harbor's task schema has
`network_mode`, with a `no-network` value and an `allowed_hosts` allowlist, and
`EnvironmentConfig` separately has `allow_internet`. Neither `gke.py` nor
`singularity/singularity.py` reads either one -- zero references in both.
Setting them would read as controls while enforcing nothing.
`10-networkpolicy.yaml` is the enforcement.

**`gke.py`'s builder.** It builds task images with Cloud Build, which does not
exist on a self-hosted cluster. It skips building entirely when `task.toml`
carries a `docker_image`, so the task is pinned to a published image and no
builder is needed:

    docker_image = "jefzda/sweap-images@sha256:<digest>"

Those images are official -- `scaleapi/SWE-bench_Pro-os` names the namespace
itself -- but every task references them by tag and upstream documents no
immutability guarantee. `eval/scripts/bake_harbor_sifs.py:resolve_digest`
resolves a tag to the digest a pull would get now; use it to write the pin.

## The sequence

The task container starts with `args: ["sleep", "infinity"]` and Harbor drives
it by exec, so preparation is an exec rather than an init container -- no
volume, no copying a few hundred megabytes per task.

1. **Pod** from the pinned digest, labelled `app.kubernetes.io/component: task`,
   with no service account token. Denied all ingress and egress.
2. **Exec: prepare.** `eval/scripts/swebenchpro_prepare_repo.sh <repo>
   <base_commit>` moves the tree to the commit the task is posed at and makes
   the upstream fix unreachable. Both halves matter; see below.
3. **Exec: agent.** Terminus-2, whose loop runs in the harness and reaches the
   model over litellm, sending the container only shell commands.
4. **Exec: verify.** The task's own tests.

## Why step 2 is not optional

The published images carry the whole upstream repository. On the image measured
here, `git reset --hard <base_commit>` leaves 67,370 commits present against the
51,852 reachable from the base commit, plus 61 remote branches and 636 tags, so
`git log --all` shows the fix by title. Upstream reports this as git reward
hacking and reproduced it under `--network none`: the network is not what leaks
it.

For a paired measurement both arms would get the same shortcut, so recovery
survives. What does not survive is sensitivity -- `git log --all | grep -i fix`
is an easy task, both arms do it, the scores compress, and the suite stops
discriminating between them. That is the entire reason to add an agentic suite,
so the strip is the difference between a measurement and a formality.

The prep does not delete `.git`. The verifier is git-shaped: `tests/config.json`
carries the patch and `solution/solve.sh` writes a diff to apply, so a repo that
cannot run `git apply` cannot be graded.

## The agent choice is load-bearing

Terminus-2 keeps model traffic in the harness. The `installed/*` agents --
codex, openhands, qwen_code, swe_agent, kimi -- are uploaded into the task
container and handed `OPENAI_BASE_URL`, which would require opening egress on
every task pod. Changing the agent means revisiting `10-networkpolicy.yaml`.

## Where step 2 hangs

`EnvironmentConfig` has no setup or pre-agent field, but it has `healthcheck`,
documented as "runs a command repeatedly after environment start to verify
readiness. All retries must pass before agent setup begins." That is a shell
command in the container, gated ahead of the agent, so no patch to `gke.py` is
needed:

    [environment.healthcheck]
    command = "/staging/swebenchpro_prepare_repo.sh /app <base_commit>"
    timeout_sec = 300

It is a readiness probe being used to mutate, which is why the script is
repeat-safe: it marks the repository and returns early on any later pass. Without
that, a retry firing after the agent had started would run `git clean -fd` over
the agent's work and the run would grade a pristine tree.

## Layout

`base/` denies task pods everything and names no addresses. `overlay/` is where
the deployment goes -- the model endpoint the harness may reach, and how the
control plane is named on a given cluster. The filled-in overlay is gitignored;
see `overlay/README.md`.

    kubectl apply -k eval/k8s/swebench-pro/overlay

Two properties of the target cluster decide the overlay, and both are worth
checking rather than assuming. Whether it is dual-stack, because a policy
written as CIDRs on one family leaves the other open -- which is why the base
denies with an empty spec instead. And which CNI enforces policy, because
naming the control plane by entity rather than by address only works on some.

## Still open

- The harness needs API credentials. Task pods run with no service account
  token; the harness cannot, since it drives them through the exec API, so it
  needs a service account with `pods/exec` in this namespace and nothing else.
