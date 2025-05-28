package edu.illinois.abhayp4.projectgenesis.cerebrum.channels;

import com.fasterxml.jackson.annotation.JsonIgnoreType;

import java.util.Arrays;

@JsonIgnoreType
public record TransmissionMessage(
    String message,
    double[] latentVector,
    int neuronId,
    long targetStep
) implements Comparable<TransmissionMessage> {
    @Override
    public int compareTo(TransmissionMessage other) {
        if (targetStep == other.targetStep) {
            if (neuronId == other.neuronId) {
                return Arrays.compare(latentVector, other.latentVector);
            }
            return Long.compare(neuronId, other.neuronId);
        }
        return Long.compare(targetStep, other.targetStep);
    }
}
