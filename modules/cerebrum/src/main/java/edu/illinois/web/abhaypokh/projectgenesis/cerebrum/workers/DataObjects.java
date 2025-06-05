package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.workers;

import com.fasterxml.jackson.annotation.JsonProperty;

public final class DataObjects {
    private DataObjects() {}

    public record ModelInput(
        @JsonProperty("Operation") String operation,
        @JsonProperty("Data") Object data
    ) {}

    public record ModelTransmissionOutput(
        @JsonProperty("Message") String message,
        @JsonProperty("LatentVector") double[] latentVector,
        @JsonProperty("TransmissionScores") double[] transmissionScores,
        @JsonProperty("TransmissionDelays") double[] transmissionDelays
    ) {}
}
