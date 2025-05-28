package edu.illinois.abhayp4.projectgenesis.cerebrum.graphs;

import com.fasterxml.jackson.annotation.JsonProperty;

final class FractalTreeGraph extends TreeGraph {
    public FractalTreeGraph(
        @JsonProperty("Tag") String tag,
        @JsonProperty("M") int m,
        @JsonProperty("Height") int height
    ) {
        super(tag, m, height);

        if (m < 2 || height < 2) {
            throw new IllegalArgumentException("m must be at least 2 and height must be at least 2");
        }

        GraphNode root = addLeafNode();
        for (int i = 0; i < m; i++) {
            addLeafNode().setAdjacentNode(root);
        }
        buildTree(m - 1, height - 1);
    }
}