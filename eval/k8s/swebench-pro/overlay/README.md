# Overlay

The base denies task pods everything and names no addresses. This is where the
deployment goes: which endpoint the harness may reach for the model, and how the
control plane is named on this particular cluster.

    cp harness-egress.yaml.example harness-egress.yaml
    cp kustomization.yaml.example  kustomization.yaml
    # fill in the endpoint, then
    kubectl apply -k .

`harness-egress.yaml` and `kustomization.yaml` are gitignored. Keep them that
way: an address on somebody's network is not part of the protocol, and this
repository is pushed.
