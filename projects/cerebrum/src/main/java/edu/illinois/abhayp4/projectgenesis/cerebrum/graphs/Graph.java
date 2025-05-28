package edu.illinois.abhayp4.projectgenesis.cerebrum.graphs;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import jakarta.annotation.Nonnull;

@JsonTypeInfo(
    use = JsonTypeInfo.Id.MINIMAL_CLASS,
    include = JsonTypeInfo.As.PROPERTY
)
@JsonSubTypes({
    @JsonSubTypes.Type(value = PerfectTreeGraph.class),
    @JsonSubTypes.Type(value = FractalTreeGraph.class),
    @JsonSubTypes.Type(value = CompleteGraph.class),
    @JsonSubTypes.Type(value = WalkGraph.class),
    @JsonSubTypes.Type(value = CycleGraph.class),
    @JsonSubTypes.Type(value = WheelGraph.class),
    @JsonSubTypes.Type(value = CustomGraph.class)
})
public sealed abstract class Graph implements Iterable<GraphNode>
    permits TreeGraph, WalkGraph, CompleteGraph, CustomGraph
{
    public final @JsonProperty("Tag") @Nonnull String tag;
    protected final List<GraphNode> nodes;

    protected Graph(String tag) {
        this.tag = tag;
        nodes = new ArrayList<>();
    }

    protected final GraphNode addNode() {
        GraphNode node = new GraphNode(nodes.size() + 1);
        nodes.add(node);
        return node;
    }

    public final @Nonnull Iterator<GraphNode> iterator() {
        return nodes.iterator();
    }

    @Override
    public final @Nonnull String toString() {
        return getClass().getSimpleName() + ": " + nodes.toString();
    }

    @Override
    public final boolean equals(Object other) {
        if (other instanceof Graph) {
            return nodes.equals(((Graph) other).nodes);
        }
        else {
            return false;
        }
    }
}