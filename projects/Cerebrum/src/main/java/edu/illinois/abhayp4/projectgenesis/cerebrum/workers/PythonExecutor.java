package edu.illinois.abhayp4.projectgenesis.cerebrum.workers;

import jakarta.annotation.Nonnull;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.ZonedDateTime;
import java.util.Arrays;
import java.util.Objects;

sealed abstract class PythonExecutor implements Closeable permits PythonClient {
    private static final String pythonExec, workerScript;

    static {
        try {
            Path tempDir = Files.createTempDirectory("python");
            new File(tempDir.toString()).deleteOnExit();

            String[] resources = {"worker.py", "requirements.txt"};

            for (String resource : resources) {
                try (PrintWriter writer = new PrintWriter(Paths.get(tempDir.toString(), resource).toString())) {
                    try (
                        BufferedReader reader = new BufferedReader(
                            new InputStreamReader(
                                Objects.requireNonNull(
                                    PythonExecutor.class.getResourceAsStream("/" + resource))))
                    ) {
                        while (reader.ready()) {
                            writer.println(reader.readLine());
                        }
                    }
                }
            }

            workerScript = Paths.get(tempDir.toString(), "worker.py").toString();
            String requirements = Paths.get(tempDir.toString(), "requirements.txt").toString();

            System.out.println("Python temporary directory: " + tempDir);

            if (!Files.isDirectory(Paths.get("output", ".venv"))) {
                runSafeCommand("python3", "-m", "venv", "output/.venv");
            }

            if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                pythonExec = Paths.get("output", ".venv", "Scripts", "python3.exe").toString();
            } else {
                pythonExec = Paths.get("output", ".venv", "bin", "python3").toString();
            }

            if (!Files.exists(Paths.get("output", ".venv", "initialized.toml"))) {
                runSafeCommand(pythonExec, "-m", "pip", "install", "--upgrade", "pip");
                runSafeCommand(pythonExec, "-m", "pip", "install", "-r", requirements);
                try (PrintWriter pw = new PrintWriter(new File("output/.venv/initialized.toml"))) {
                    pw.println("[initialized]");
                    pw.printf("project = \"%s\"%n", "cerebrum");
                    pw.printf("time = \"%s\"%n", ZonedDateTime.now());
                }
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private final Thread thread;

    protected PythonExecutor(@Nonnull String argument) {
        thread = new Thread(() -> runSafeCommand(pythonExec, workerScript, argument));
        thread.start();
    }

    @Override
    public void close() {
        try {
            thread.join();
        }
        catch (InterruptedException e) {
            throw new RuntimeException("Cannot interrupt during Python process end");
        }
    }

    private static void runSafeCommand(@Nonnull String... command) {
        try {
            ProcessBuilder pb = new ProcessBuilder(command).inheritIO();
            Process process = pb.start();
            process.waitFor();
            if (process.exitValue() != 0) {
                throw new RuntimeException("Command " + Arrays.asList(command) + " failed");
            }
        } catch (Exception e) {
            System.err.println(e.getMessage());
            System.exit(1);
        }
    }
}
