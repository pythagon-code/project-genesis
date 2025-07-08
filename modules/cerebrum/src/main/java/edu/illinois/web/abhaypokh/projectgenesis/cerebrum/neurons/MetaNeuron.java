package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.neurons;

import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.channels.TransmissionMessage;
import jakarta.annotation.Nonnull;

import java.util.ArrayList;
import java.util.List;

sealed class MetaNeuron extends RelayNeuron permits ResponseNeuron {
    private final List<RelayNeuron> subNeuronList;
    private final int level;

    public MetaNeuron(int level) {
        this.level = level;
        subNeuronList = new ArrayList<>();
    }

    public void addSubNeuron(@Nonnull RelayNeuron subNeuron) {
        if (subNeuronList.contains(subNeuron)) {
            throw new IllegalArgumentException();
        }

        subNeuronList.add(subNeuron);
    }

    @Override
    protected void onMessageReceived(@Nonnull TransmissionMessage message) {

    }

    @Override
    protected void onAwake() {

    }
}
