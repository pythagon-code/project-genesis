package edu.illinois.web.abhaypokh.projectgenesis.cerebrum.misc;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

import com.fasterxml.jackson.core.*;
import com.fasterxml.jackson.annotation.*;
import com.fasterxml.jackson.databind.*;

import edu.illinois.web.abhaypokh.projectgenesis.cerebrum.channels.PriorityMessageChannel;

public class JsonTest {
    @JsonAutoDetect(fieldVisibility = JsonAutoDetect.Visibility.ANY)
    private static class MyClass1 {
        private final int i, j, k;
        public final int l;
        public PriorityMessageChannel channel;

        @JsonCreator
        public MyClass1(
            @JsonProperty("i") int i,
            @JsonProperty("j") int j,
            @JsonProperty("k") int k
        ) {
            this.i = i;
            this.j = j;
            this.k = j;
            this.l = 4;
            channel = null;
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof MyClass1 obj)) {
                return false;
            }

            return (i == obj.i) && (j == obj.j) && (k == obj.k) && (channel == obj.channel);
        }
    }
    
    @Test
    public void testJackson1() throws JsonProcessingException {
        ObjectMapper objectMapper = new ObjectMapper();
        MyClass1 obj = new MyClass1(1, 2, 3);
        obj.channel = new PriorityMessageChannel();
        String json = objectMapper.writeValueAsString(obj);
        MyClass1 obj2 = objectMapper.readValue(json, MyClass1.class);
        assertNotEquals(obj, obj2);
        obj.channel = null;
        assertEquals(obj, obj2);
        assertEquals(4, obj.l);

        System.out.println("testJackson1(): ");
        System.out.println(json);
        System.out.println();
    }

    private static class MyClass2 {
        public final int i;

        @JsonCreator
        public MyClass2(
            @JsonProperty("i") int i
        ) {
            throw new RuntimeException("Should not be called.");
        }

        protected MyClass2(int i, int __) {
            this.i = i;
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof MyClass2)) {
                return false;
            }

            MyClass2 obj = (MyClass2) other;
            return (i == obj.i);
        }
    }

    private static class MyClass2Child extends MyClass2 {
        @JsonProperty("J") final private int j = 0;

        @JsonCreator
        public MyClass2Child(
            @JsonProperty("i") int i,
            @JsonProperty("J") int j
        ) {
            super(i, 0);
        }
    }
    
    @Test
    public void testJacksonSubclass1() throws JsonProcessingException {
        ObjectMapper objectMapper = new ObjectMapper();
        MyClass2Child obj = new MyClass2Child(10, 20);
        String json = objectMapper.writeValueAsString(obj);
        MyClass2Child obj2 = objectMapper.readValue(json, MyClass2Child.class);
        assertEquals(obj, obj2);

        System.out.println("testJacksonSubclass1(): ");
        System.out.println(json);
        System.out.println();
    }

    public record RecordClass1(
        @JsonProperty("I") int i,
        @JsonProperty("String") String str
    ) { }

    @Test
    public void testJacksonRecordClass1() throws JsonProcessingException {
        ObjectMapper objectMapper = new ObjectMapper();
        RecordClass1 obj = new RecordClass1(10, "hello");
        String json = objectMapper.writeValueAsString(obj);
        RecordClass1 obj2 = objectMapper.readValue(json, RecordClass1.class);
        assertEquals(obj, obj2);

        System.out.println("testJacksonRecordClass1(): ");
        System.out.println(json);
        System.out.println(obj);
        System.out.println();
    }

    @Test
    public void testJacksonReadPrimitives1() throws JsonProcessingException {
        ObjectMapper objectMapper = new ObjectMapper();
        int i = objectMapper.readValue("10", int.class);
        assertEquals(10, i);
        System.out.println("testJacksonReadPrimitives1(): ");
        System.out.println(i);
        System.out.println();
    }
}