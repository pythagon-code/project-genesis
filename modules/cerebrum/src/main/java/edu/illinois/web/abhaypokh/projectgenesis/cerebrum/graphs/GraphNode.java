package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.graphs;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreType;
import jakarta.annotation.Nonnull;

@JsonIgnoreType
public final class GraphNode {
    public final int id;
    private final List<GraphNode> adjacentNodes;

    public GraphNode(int id) {
        this.id = id;
        adjacentNodes = new ArrayList<>();
    }

    public @Nonnull Iterable<GraphNode> getAdjacentNodes() {
        return Collections.unmodifiableCollection(adjacentNodes);
    }

    void setAdjacentNode(GraphNode node) {
        if (adjacentNodes.contains(node)) {
            return;
        }
        adjacentNodes.add(node);
        node.adjacentNodes.add(this);
    }

    @Override
    public @Nonnull String toString() {
        return "GraphNode[id=" + id + "]";
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof GraphNode otherNode) {
            return id == otherNode.id && adjacentNodes.equals(otherNode.adjacentNodes);
        } else {
            return false;
        }
    }
}