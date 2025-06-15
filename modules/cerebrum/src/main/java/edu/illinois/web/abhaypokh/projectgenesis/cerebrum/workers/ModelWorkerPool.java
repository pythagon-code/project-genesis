package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.workers;

import java.io.Closeable;

import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

public final class ModelWorkerPool implements Closeable {
    private final int maxNeuronsPerWorker;
    private final Map<ModelWorker, Integer> workerUsage;
    private final List<ModelWorker> availableWorkers;
    private final Random random;

    public ModelWorkerPool(int neuronCount, int workerCount, int randomSeed) {
        maxNeuronsPerWorker = Math.ceilDiv(neuronCount, workerCount);
        workerUsage = new HashMap<>();

        for (int i = 0; i < workerCount; i++) {
            workerUsage.put(new ModelWorker(), 0);
        }

        availableWorkers = new ArrayList<>(workerUsage.keySet());
        random = new Random(randomSeed);
    }

    @SuppressWarnings("resource")
    public ModelWorker getAvailableWorker() {
        if (availableWorkers.isEmpty()) {
            throw new NoSuchElementException();
        }

        int workerIdx = random.nextInt(availableWorkers.size());
        ModelWorker worker = availableWorkers.get(workerIdx);

        int usage = workerUsage.get(worker) + 1;
        workerUsage.put(worker, usage);

        if (usage == maxNeuronsPerWorker) {
            availableWorkers.remove(workerIdx);
        }

        return worker;
    }

    @Override
    public void close() {
        for (ModelWorker worker : workerUsage.keySet()) {
            worker.close();
        }
    }

    public static void nothing() {
        ModelWorker.nothing();
    }
}