package edu.illinois.abhayp4.projectgenesis.cerebrum.graphs;

import com.fasterxml.jackson.annotation.JsonProperty;


final class PerfectTreeGraph<T> extends TreeGraph {
    public PerfectTreeGraph(
        @JsonProperty("Tag") String tag,
        @JsonProperty("M") int m,
        @JsonProperty("Height") int height
    ) {
        super(tag, m, height);

        if (m < 2 || height < 1) {
            throw new IllegalArgumentException("m must be at least 2 and height must be positive");
        }

        addLeafNode();
        buildTree(m, height);
    }
}