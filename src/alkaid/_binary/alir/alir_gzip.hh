#include <cstddef>
#include <stdexcept>
#include <string>

#include <zlib.h>

namespace alir {

    inline bool is_gzip_magic(const unsigned char *data, size_t size) {
        return size >= 2 && data[0] == 0x1f && data[1] == 0x8b;
    }
    inline bool is_gzip_magic(const char *data, size_t size) {
        return is_gzip_magic(reinterpret_cast<const unsigned char *>(data), size);
    }

    inline std::string gzip_inflate(const char *data, size_t size) {
        z_stream stream{};
        if (inflateInit2(&stream, 15 + 32) != Z_OK)
            throw std::runtime_error("zlib inflateInit2 failed");

        stream.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(data));
        stream.avail_in = static_cast<uInt>(size);

        constexpr size_t CHUNK = 64 * 1024;
        std::string out;
        out.reserve(size * 16);
        char buf[CHUNK];
        int ret = Z_OK;
        while (ret != Z_STREAM_END) {
            stream.next_out = reinterpret_cast<Bytef *>(buf);
            stream.avail_out = CHUNK;
            ret = inflate(&stream, Z_NO_FLUSH);
            if (ret != Z_OK && ret != Z_STREAM_END) {
                inflateEnd(&stream);
                throw std::runtime_error("zlib inflate failed with code " + std::to_string(ret));
            }
            out.append(buf, CHUNK - stream.avail_out);
            if (stream.avail_in == 0 && ret != Z_STREAM_END) {
                inflateEnd(&stream);
                throw std::runtime_error("zlib inflate: truncated input");
            }
        }
        inflateEnd(&stream);
        return out;
    }

} // namespace alir
