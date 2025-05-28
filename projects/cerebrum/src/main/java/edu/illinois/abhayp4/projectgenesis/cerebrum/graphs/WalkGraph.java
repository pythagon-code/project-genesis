package edu.illinois.abhayp4.projectgenesis.cerebrum.graphs;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

sealed class WalkGraph extends Graph permits CycleGraph {
    @JsonProperty("N") protected final int n;

    @JsonCreator
    public WalkGraph(
        @JsonProperty("Tag") String tag,
        @JsonProperty("N") int n
    ) {
        super(tag);

        if (n < 2) {
            throw new IllegalArgumentException("N must be positive");
        }

        this.n = n;

        addNode();
        for (int i = 1; i < n; i++) {
            addNode().setAdjacentNode(nodes.getLast());
        }
    }
}