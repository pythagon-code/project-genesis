plugins {
    application
    id("org.beryx.jlink") version "3.1.1"
}

group = "edu.illinois.abhayp4.projectgenesis.neweden"
version = "1.0-SNAPSHOT"

application {
    mainModule.set("edu.illinois.abhayp4.projectgenesis.neweden")
    mainClass.set("edu.illinois.abhayp4.projectgenesis.neweden.main.Main")
}

jlink {
    options.set(listOf("--no-header-files", "--no-man-pages"))

    launcher {
        name = "project-genesis-new-eden"
    }
}