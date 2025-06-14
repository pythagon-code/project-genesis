package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.graphs;

import com.fasterxml.jackson.annotation.JsonProperty;
import jakarta.annotation.Nonnull;

import java.util.LinkedList;
import java.util.List;

sealed abstract class TreeGraph extends Graph permits PerfectTreeGraph, FractalTreeGraph {
    @JsonProperty("M") protected final int m;
    @JsonProperty("Height") protected final int height;

    private final List<GraphNode> leafNodes;

    protected TreeGraph(
        @JsonProperty("Tag") String tag,
        @JsonProperty("M") int m,
        @JsonProperty("Height") int height
    ) {
        super(tag);
        this.m = m;
        this.height = height;
        leafNodes = new LinkedList<>();
    }

    protected final @Nonnull GraphNode addLeafNode() {
        GraphNode node = addNode();
        leafNodes.add(node);
        return node;
    }

    protected final void buildTree(int childrenPerParent, int depth) {
        for (int k = 0; k < depth; k++) {
            int size = leafNodes.size();
            for (int i = 0; i < size; i++) {
                GraphNode parent = leafNodes.removeFirst();
                for (int j = 0; j < childrenPerParent; j++) {
                    addLeafNode().setAdjacentNode(parent);
                }
            }
        }
    }
}