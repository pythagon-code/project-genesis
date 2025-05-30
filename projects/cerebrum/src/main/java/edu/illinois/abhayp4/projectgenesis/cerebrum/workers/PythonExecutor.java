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
                        String line;
                        while ((line = reader.readLine()) != null) {
                            writer.println(line);
                        }
                    }
                }
            }

            workerScript = Paths.get(tempDir.toString(), "worker.py").toString();
            String requirements = Paths.get(tempDir.toString(), "requirements.txt").toString();

            System.out.println("Python temporary directory: " + tempDir);

            if (!Files.isDirectory(Paths.get("output", ".venv"))) {
                runSafeCommandAndWait("python", "-m", "venv", "output/.venv");
            }

            if (System.getProperty("os.name").toLowerCase().contains("windows")) {
                pythonExec = Paths.get("output", ".venv", "Scripts", "python.exe").toString();
            } else {
                pythonExec = Paths.get("output", ".venv", "bin", "python").toString();
            }

            if (!Files.exists(Paths.get("output", ".venv", "initialized.toml"))) {
                runSafeCommandAndWait(pythonExec, "-m", "pip", "install", "--upgrade", "pip");
                runSafeCommandAndWait(pythonExec, "-m", "pip", "install", "-r", requirements);
                try (PrintWriter pw = new PrintWriter("output/.venv/initialized.toml")) {
                    pw.println("[initialized]");
                    pw.println("project = \"cerebrum\"");
                    pw.println("time = \"" + ZonedDateTime.now() + "\"");
                }
            }

        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private Process process = null;

    protected final void startProcess(@Nonnull String... command) {
        if (process != null) {
          throw new RuntimeException("Process already started");
        }

        process = runSafeCommand(command);
    }

    @Override
    public void close() {
        if (process == null) {
            throw new RuntimeException("Process never started");
        }

        try {
            process.waitFor();
        }
        catch (InterruptedException e) {
            throw new RuntimeException("Cannot interrupt during Python process end");
        }
    }

    private static void runSafeCommandAndWait(@Nonnull String... command) {
        try {
            Process process = runSafeCommand(command);
            process.waitFor();
            if (process.exitValue() != 0) {
                throw new RuntimeException("Command " + Arrays.asList(command) + " failed");
            }
        } catch (InterruptedException e) {
            throw new RuntimeException("Cannot interrupt during Python process startProcess");
        }
    }

    private static @Nonnull Process runSafeCommand(@Nonnull String... command) {
        try {
            ProcessBuilder pb = new ProcessBuilder(command).inheritIO();
            return pb.start();
        } catch (IOException e) {
            System.err.println(e.getMessage());
            System.exit(1);
            return null;
        }
    }
}
