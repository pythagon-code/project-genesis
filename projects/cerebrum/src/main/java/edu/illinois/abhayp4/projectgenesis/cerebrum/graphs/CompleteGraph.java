package edu.illinois.abhayp4.projectgenesis.cerebrum.graphs;

import com.fasterxml.jackson.annotation.JsonProperty;

final class CompleteGraph extends Graph {
    @JsonProperty("N") private final int n;

    public CompleteGraph(
        @JsonProperty("Tag") String tag,
        @JsonProperty("N") int n
    ) {
        super(tag);

        if (n < 4) {
            throw new IllegalArgumentException("n must be at least 4");
        }

        this.n = n;

        for (int i = 0; i < n; i++) {
            addNode();
            for (int j = 0; j < i; j++) {
                nodes.get(i).setAdjacentNode(nodes.get(j));
            }
        }
    }
}