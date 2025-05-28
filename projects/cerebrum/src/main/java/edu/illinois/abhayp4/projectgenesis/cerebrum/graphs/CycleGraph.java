package edu.illinois.abhayp4.projectgenesis.cerebrum.graphs;

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.annotation.Nonnull;

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