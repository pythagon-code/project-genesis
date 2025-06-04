module edu.illinois.web.abhaypokh.projectgenesis.cerebrum {
    requires com.fasterxml.jackson.databind;
    requires jakarta.annotation;
    requires java.logging;
    requires javafx.controls;
    requires javafx.fxml;
    requires javafx.graphics;
    requires org.slf4j;
    requires org.yaml.snakeyaml;

    exports edu.illinois.web.abhaypokh.projectgenesis.cerebrum.api;
    opens edu.illinois.web.abhaypokh.projectgenesis.cerebrum.application to javafx.fxml, javafx.graphics;
}