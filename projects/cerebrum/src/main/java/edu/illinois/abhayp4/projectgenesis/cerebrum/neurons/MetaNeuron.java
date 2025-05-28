package edu.illinois.abhayp4.projectgenesis.cerebrum.neurons;

import edu.illinois.abhayp4.projectgenesis.cerebrum.channels.TransmissionMessage;
import jakarta.annotation.Nonnull;

import java.util.ArrayList;
import java.util.List;

sealed class MetaNeuron extends RelayNeuron permits ResponseNeuron {
    private List<RelayNeuron> neurons;
    
    public MetaNeuron() {
        neurons = new ArrayList<>();
    }

    public void setAdjacentNeuron(RelayNeuron other) {
        
    }

    @Override
    protected void onMessageReceived(TransmissionMessage message) {
        // TODO Auto-generated method stub
        
    }

    @Override
    protected void onAwaken() {
        // TODO Auto-generated method stub
        
    }
}
