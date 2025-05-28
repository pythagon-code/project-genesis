package edu.illinois.abhayp4.projectgenesis.cerebrum.graphs;

import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonIgnoreType;
import jakarta.annotation.Nonnull;

@JsonIgnoreType
public final class GraphNode {
    private int id;
    private final List<GraphNode> adjacentNodes;

    public GraphNode(int id) {
        this.id = id;
        adjacentNodes = new ArrayList<>();
    }

    public Iterable<GraphNode> getAdjacentNodes() {
        return adjacentNodes;
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
        return "" + id + ": " + adjacentNodes;
    }

    @Override
    public boolean equals(Object other) {
        if (other instanceof GraphNode) {
            return adjacentNodes.equals(other) && (id == ((GraphNode) other).id);
        }
        else {
            return false;
        }
    }
}