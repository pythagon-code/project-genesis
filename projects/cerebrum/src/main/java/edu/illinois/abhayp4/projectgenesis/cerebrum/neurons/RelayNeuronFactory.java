/**
 * RelayNeuronFactory.java
 * @author Abhay Pokhriyal
 */

package edu.illinois.abhayp4.projectgenesis.cerebrum.neurons;

import java.io.Closeable;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import edu.illinois.abhayp4.projectgenesis.cerebrum.brain.SimulatorConfig;
import jakarta.annotation.Nonnull;

public class RelayNeuronFactory implements Closeable {
    @JsonProperty("Neurons") private final List<RelayNeuron> neurons;
    @JsonProperty("SimulatorConfig") public final @Nonnull SimulatorConfig simulatorConfig;

    @JsonCreator
    private RelayNeuronFactory(
        @JsonProperty("Neurons") @Nonnull List<RelayNeuron> neurons,
        @JsonProperty("SimulatorConfig") @Nonnull SimulatorConfig simulatorConfig
    ) {
        this.neurons = neurons;
        this.simulatorConfig = simulatorConfig;
    }

    public RelayNeuronFactory(@Nonnull SimulatorConfig simulatorConfig) {
        this.neurons = new ArrayList<>();
        this.simulatorConfig = simulatorConfig;
    }

    public void build() {
        
    }

    @Override
    public void close() {

    }
}