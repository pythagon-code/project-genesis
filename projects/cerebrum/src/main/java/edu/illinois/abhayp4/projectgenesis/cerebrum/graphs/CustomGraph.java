package edu.illinois.abhayp4.projectgenesis.cerebrum.graphs;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

public final class CustomGraph extends Graph {
    @JsonProperty("NVertices") public int nVertices;
    @JsonProperty("Edges") public List<List<Integer>> edges;

    public CustomGraph(
        @JsonProperty("Tag") String tag,
        @JsonProperty("NVertices") int nVertices,
        @JsonProperty("Edges") List<List<Integer>> edges
    ) {
        super(tag);
        this.nVertices = nVertices;
        this.edges = edges;
    }
}