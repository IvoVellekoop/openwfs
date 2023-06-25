default_vertex_shader = """
        #version 440 core
        layout(location = 0) in vec2 slm_coordinate;
        layout(location = 1) in vec2 texture_coordinate;
        layout(binding = 1, std140) uniform globals {
            mat3x2 transform;
        };
        out vec2 texCoord;

        void main() {
            vec2 pos = transform * vec3(slm_coordinate, 1.0);
            gl_Position = vec4(pos.x, pos.y, 0.0, 1.0);
            texCoord = texture_coordinate;
        }
        """

default_fragment_shader = """
        #version 440 core
        in vec2 texCoord;
        out vec4 colorOut;
        layout(binding = 0) uniform sampler2D texSampler;
        
        void main() {
            float val = texture(texSampler, texCoord).r;
            colorOut = vec4(val, val, val, 1.0);
        }
    """

post_process_fragment_shader = """
        #version 440 core
        in vec2 texCoord;
        out vec4 colorOut;
        layout(binding = 0) uniform sampler2D texSampler;
        layout(binding = 1) uniform sampler1D LUT;
        const float scale = 0.1591549431f; // = 1 / 2pi

        void main() {
            float raw = texture(texSampler, texCoord).r * scale;
            float val = raw;//texture(LUT, raw).r;
            colorOut = vec4(val, val, val, 1.0);
        }
    """