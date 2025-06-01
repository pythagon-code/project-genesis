package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.graphs;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

sealed class CycleGraph extends WalkGraph permits WheelGraph {
    @JsonCreator
    public CycleGraph(
        @JsonProperty("Tag") String tag,
        @JsonProperty("N") int n
    ) {
        super(tag, n - 1);

        if (n < 3) {
            throw new IllegalArgumentException("CycleGraph must have at least 3 nodes");
        }
        
        nodes.getFirst().setAdjacentNode(nodes.getLast());
    }
}