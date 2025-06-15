package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.channels;

import com.fasterxml.jackson.annotation.JsonIgnoreType;

import java.util.Arrays;

@JsonIgnoreType
public record TransmissionMessage(
    String message,
    double[] latentVector,
    int senderId,
    long targetStep
) implements Comparable<TransmissionMessage> {
    @Override
    public int compareTo(TransmissionMessage other) {
        if (targetStep == other.targetStep) {
            if (senderId == other.senderId) {
                return Arrays.compare(latentVector, other.latentVector);
            }
            return Long.compare(senderId, other.senderId);
        }
        return Long.compare(targetStep, other.targetStep);
    }
}
