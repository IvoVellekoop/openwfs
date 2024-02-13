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

# This shader is used to apply a software lookup table to phase image.
# The phase in radians is converted to a coordinate in the lookup table
# using the equation: texCoord = phase / (2 * pi) + 0.5 / 256
# where the offset is used for proper rounding, so that with the standard lookup table
# the range -δ to δ maps to a gray value 0 instead of
# negative values mapping to 255 and positive values mapping to 0.
# Since the lookup table texture is configured to use GL_WRAP,
# only the fractional part of texCoord is used (i.e., texCoord - floor(texCoord)).
#
post_process_fragment_shader = """
        #version 440 core
        in vec2 texCoord;
        out vec4 colorOut;
        layout(binding = 0) uniform sampler2D texSampler;
        layout(binding = 1) uniform sampler1D LUT;
        const float scale = 0.15915494309189535f; // corresponds to 1/(2 pi).
        const float offset = 0.001953125f; // corresponds to 0.5/256.

        void main() {
            float raw = texture(texSampler, texCoord).r * scale + offset;
            float val = texture(LUT, raw).r;
            colorOut = vec4(val, val, val, 1.0);
        }
    """

post_process_vertex_shader = """
        #version 440 core
        layout(location = 0) in vec2 slm_coordinate;
        layout(location = 1) in vec2 texture_coordinate;
        out vec2 texCoord;

        void main() {
            gl_Position = vec4(slm_coordinate, 0.0, 1.0);
            texCoord = texture_coordinate;
        }
    """
