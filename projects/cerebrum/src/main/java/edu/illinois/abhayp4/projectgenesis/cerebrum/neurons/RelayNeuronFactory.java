/**
 * RelayNeuronFactory.java
 * @author Abhay Pokhriyal
 */

package edu.illinois.abhayp4.projectgenesis.cerebrum.neurons;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import edu.illinois.abhayp4.projectgenesis.cerebrum.brain.SimulatorConfig;
import jakarta.annotation.Nonnull;

public class RelayNeuronFactory {
    @JsonProperty("Neurons") private /*final*/ @Nonnull List<RelayNeuron> neurons;
    @JsonProperty("SimulatorConfig") public /*final*/ @Nonnull SimulatorConfig simulatorConfig;

    //private final Map<Graph, List<RelayNeuron>> adjacency;

    @JsonCreator
    private RelayNeuronFactory(
        @JsonProperty("Neurons") List<RelayNeuron> neurons,
        @JsonProperty("SimulatorConfig") SimulatorConfig simulatorConfig
    ) {
        this.neurons = neurons;
        this.simulatorConfig = simulatorConfig;

        //adjacency = new HashMap<>();

    }

    public RelayNeuronFactory(SimulatorConfig simulatorConfig) {

    }

    public void build() {
        
    }
}