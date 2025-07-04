import java.nio.file.Paths

plugins {
    application
    id("org.openjfx.javafxplugin") version "0.1.0"
}

group = "edu.illinois.web.abhaypokh.projectgenesis"
version = "1.0-SNAPSHOT"

allprojects {
    apply(plugin = "java")
    apply(plugin = "application")

    java {
        toolchain {
            languageVersion.set(JavaLanguageVersion.of(23))
            vendor.set(JvmVendorSpec.ADOPTIUM)
        }
        modularity.inferModulePath = true
    }

    repositories {
        mavenCentral()
    }

    dependencies {
        testImplementation("org.junit.jupiter:junit-jupiter:5.7.1")
        testRuntimeOnly("org.junit.platform:junit-platform-launcher")
        implementation("jakarta.annotation:jakarta.annotation-api:2.1.1")
        implementation(platform("org.apache.logging.log4j:log4j-bom:2.12.4"))
        implementation("org.apache.logging.log4j:log4j-api")
        implementation("org.apache.logging.log4j:log4j-core")
    }

    tasks.withType<Test>().configureEach {
        useJUnitPlatform()
        maxHeapSize = "1G"

        testLogging {
            events("passed", "failed", "skipped")
            showStandardStreams = true
        }
    }

    tasks.withType<ProcessResources> {
        duplicatesStrategy = DuplicatesStrategy.EXCLUDE
    }

    tasks.register<Copy>("copyDeps") {
        dependsOn(tasks.build)

        from(configurations.runtimeClasspath)
        into(layout.buildDirectory.dir("deps/"))
    }

    tasks.register<JavaExec>("runModularJar") {
        dependsOn(tasks.getByName("copyDeps"))

        mainModule.set(application.mainModule.get())
        mainClass.set(application.mainClass.get())

        classpath = files()

        val libs = layout.buildDirectory.dir("libs/").get().asFile.absolutePath
        val deps = layout.buildDirectory.dir("deps/").get().asFile.absolutePath

        val args = arrayListOf("--module-path", "$libs${File.pathSeparator}$deps")

        if (project.properties["mode"] == "debug")
            args.add("-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,address=*:9000")

        jvmArgs = args
    }

    tasks.register<Delete>("cleanImage") { delete(layout.buildDirectory.dir("image/")) }

    tasks.register<Exec>("jlink") {
        dependsOn(tasks.getByName("copyDeps"), tasks.getByName("cleanImage"))

        var launcher = javaToolchains.launcherFor {
            languageVersion.set(JavaLanguageVersion.of(23))
            vendor.set(JvmVendorSpec.ADOPTIUM)
        }.get().metadata.installationPath.asFile.toString()

        var jlink = Paths.get(launcher, "bin", "jlink").toAbsolutePath().toString()

        var libs = layout.buildDirectory.dir("libs/").get().asFile.absolutePath
        var deps = layout.buildDirectory.dir("deps/").get().asFile.absolutePath

        commandLine = listOf(
            jlink,
            "--module-path", "$libs${File.pathSeparator}$deps",
            "--add-modules", application.mainModule.get(),
            "--output", layout.buildDirectory.dir("image/").get().asFile.absolutePath,
            "--launcher", "app_launcher=${application.mainModule.get()}/${application.mainClass.get()}",
            "--strip-debug",
            "--no-header-files",
            "--no-man-pages"
        )
    }
}

dependencies {
    implementation(project(":cerebrum"))
    implementation(project(":new-eden"))
}

application {
    mainModule.set("edu.illinois.web.abhaypokh.projectgenesis")
    mainClass.set("edu.illinois.web.abhaypokh.projectgenesis.application.Main")
}