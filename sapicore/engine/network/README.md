# Network Abstraction

Network objects own a graph representation containing ensembles as vertices and synapses as edges, with an
identifier and an object `reference` included in each element's metadata.

Synapses in the network configuration YAML are given in the format "E1->E2", where Ei is an ensemble identifier and
"->" is treated as a special splitting string throughout the code base.

Self-connections ("E2->E2") are perfectly
valid and are treated the same way, except for zeroing out the diagonal in the connection mask matrix to disable
connectivity from individual tensor elements to themselves.

While the developer may micromanage information flow by subclassing `Network`, the generic
`forward()` implementation outlined below should be sufficient for most models:

* The generic forward sweep follows multi-source breadth-first search order (BFS), starting from the root node layer
and advancing along outgoing edges (synapse collections).


* On each step, each ensemble integrates <b>all</b> of its synaptic inputs (the `output` field of the synapse object
corresponding to each incoming edge).


* The sweep continues until the entire graph has been traversed, with each node forwarded once and all outgoing
edges from every node forwarded once.
