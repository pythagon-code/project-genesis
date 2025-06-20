plugins {
    application
    id("org.openjfx.javafxplugin") version "0.1.0"
}

group = "edu.illinois.web.abhaypokh.projectgenesis.cerebrum"
version = "1.0-SNAPSHOT"

dependencies {
    implementation("org.yaml:snakeyaml:2.3")
    implementation(platform("com.fasterxml.jackson:jackson-bom:2.19.0"))
    implementation("com.fasterxml.jackson.core:jackson-core")
    implementation("com.fasterxml.jackson.core:jackson-annotations")
    implementation("com.fasterxml.jackson.core:jackson-databind")
}

sourceSets {
    main {
        resources {
            srcDirs("src/main/resources/", "src/main/python/")
        }
    }
}

application {
    mainModule.set("edu.illinois.web.abhaypokh.projectgenesis.cerebrum")
    mainClass.set("edu.illinois.web.abhaypokh.projectgenesis.cerebrum.api.CerebrumApi")
}

javafx {
    version = "24.0.1"
    modules("javafx.controls", "javafx.fxml", "javafx.graphics")
}