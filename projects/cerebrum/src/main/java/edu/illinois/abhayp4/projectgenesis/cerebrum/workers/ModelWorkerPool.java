/**
 * ModelWorkerPool.java
 * @author Abhay Pokhriyal
 */

package edu.illinois.abhayp4.projectgenesis.cerebrum.workers;

import java.io.Closeable;

import java.util.HashMap;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

public final class ModelWorkerPool implements Closeable {
    private int maxThreadsPerClient;
    private Map<ModelWorker, Integer> clientUsage;
    private List<ModelWorker> availableClients;
    private Random random;

    public ModelWorkerPool() {
        maxThreadsPerClient = Math.ceilDiv(0, 0);
        clientUsage = new HashMap<>();

        for (int i = 0; i < getNPythonWorkers(); i++) {
            clientUsage.put(new ModelWorker(), 0);
        }

        availableClients = new ArrayList<>(clientUsage.keySet());
        random = new Random();
    }

    private int getNPythonWorkers() {
        return Math.max(0, 0);
    }

    public synchronized ModelWorker getAvailableClient() {
        if (availableClients.isEmpty()) {
            throw new NoSuchElementException();
        }

        int clientIdx = random.nextInt(availableClients.size());
        ModelWorker client = availableClients.get(clientIdx);
        
        int usage = clientUsage.get(client) + 1;
        clientUsage.put(client, usage);

        if (usage == maxThreadsPerClient) {
            availableClients.remove(clientIdx);
        }

        return client;
    }

    @Override
    public void close() {
        for (ModelWorker client : clientUsage.keySet()) {
            client.close();
        }
    }
}