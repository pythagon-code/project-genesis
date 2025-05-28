package edu.illinois.abhayp4.projectgenesis.cerebrum.misc;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
import org.yaml.snakeyaml.Yaml;

import java.util.List;
import java.util.Map;

public class YamlTest {
    @Test
    public void yamlTestNestedLists1() {
        Yaml yaml = new Yaml();
        Map<String, Object> object = yaml.load("""
            test:
                - [1, 2, 3, 4, 5]
                - [6, 7, 8, 9, 10]
            """);
        assertTrue(object.get("test") instanceof List<?>);
        List<?> lists = (List<?>) object.get("test");
        assertTrue(lists.get(0) instanceof List<?>);
        List<?> list0 = (List<?>) lists.get(0);
        assertTrue(list0.get(0) instanceof Integer);

        System.out.println("yamlTestNestedLists1():");
        System.out.println(object);
    }
}
