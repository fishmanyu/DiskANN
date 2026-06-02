// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

constexpr uint64_t SECTOR_LEN = 4096;

uint64_t div_round_up(uint64_t x, uint64_t y)
{
    return (x / y) + (x % y != 0);
}

uint64_t round_up(uint64_t x, uint64_t y)
{
    return div_round_up(x, y) * y;
}

bool file_exists(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    return in.good();
}

uint64_t file_size(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Could not open " + path);
    in.seekg(0, std::ios::end);
    return static_cast<uint64_t>(in.tellg());
}

template <typename T> T read_value(std::ifstream &in)
{
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(T));
    if (!in)
        throw std::runtime_error("Unexpected end of file while reading index");
    return value;
}

void write_edges(std::ofstream &out, uint32_t node_id, const uint32_t *neighbors, uint32_t degree)
{
    for (uint32_t i = 0; i < degree; ++i)
    {
        out << node_id << ',' << neighbors[i] << '\n';
    }
}

bool looks_like_vamana_graph(const std::string &path)
{
    if (!file_exists(path))
        return false;

    std::ifstream in(path, std::ios::binary);
    if (!in)
        return false;

    const uint64_t actual_size = file_size(path);
    const uint64_t expected_size = read_value<uint64_t>(in);
    if (expected_size != actual_size || expected_size < 24)
        return false;

    const uint32_t max_degree = read_value<uint32_t>(in);
    read_value<uint32_t>(in); // start node
    read_value<uint64_t>(in); // frozen point count

    uint64_t bytes_read = 24;
    uint64_t nodes_seen = 0;
    while (bytes_read < expected_size && nodes_seen < 1024)
    {
        const uint32_t degree = read_value<uint32_t>(in);
        if (degree > max_degree || bytes_read + sizeof(uint32_t) * (uint64_t(degree) + 1) > expected_size)
            return false;
        in.seekg(sizeof(uint32_t) * uint64_t(degree), std::ios::cur);
        bytes_read += sizeof(uint32_t) * (uint64_t(degree) + 1);
        ++nodes_seen;
    }

    return bytes_read <= expected_size && nodes_seen > 0;
}

void export_vamana_graph(const std::string &path, const std::string &output_csv)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Could not open Vamana graph file " + path);

    std::ofstream out(output_csv);
    if (!out)
        throw std::runtime_error("Could not open output CSV " + output_csv);

    const uint64_t expected_size = read_value<uint64_t>(in);
    const uint32_t max_degree = read_value<uint32_t>(in);
    const uint32_t start = read_value<uint32_t>(in);
    const uint64_t frozen_points = read_value<uint64_t>(in);

    out << "node_id,neighbor_id\n";

    uint64_t bytes_read = 24;
    uint32_t node_id = 0;
    std::vector<uint32_t> neighbors;
    while (bytes_read != expected_size)
    {
        if (bytes_read > expected_size)
            throw std::runtime_error("Vamana graph file size metadata is inconsistent");

        const uint32_t degree = read_value<uint32_t>(in);
        if (degree > max_degree)
            throw std::runtime_error("Found node degree larger than graph header max degree");

        neighbors.resize(degree);
        in.read(reinterpret_cast<char *>(neighbors.data()), sizeof(uint32_t) * uint64_t(degree));
        if (!in)
            throw std::runtime_error("Unexpected end of file while reading Vamana neighbors");

        write_edges(out, node_id, neighbors.data(), degree);
        bytes_read += sizeof(uint32_t) * (uint64_t(degree) + 1);
        ++node_id;
    }

    std::cerr << "Exported " << node_id << " Vamana nodes from " << path << " to " << output_csv
              << " (start=" << start << ", frozen_points=" << frozen_points << ")." << std::endl;
}

struct DiskMetadata
{
    uint64_t nodes = 0;
    uint64_t dims = 0;
    uint64_t medoid = 0;
    uint64_t max_node_len = 0;
    uint64_t nodes_per_sector = 0;
    uint64_t frozen_points = 0;
    uint64_t frozen_location = 0;
    uint64_t reorder_exists = 0;
};

DiskMetadata read_disk_metadata(const std::string &path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Could not open disk index file " + path);

    const uint32_t metadata_rows = read_value<uint32_t>(in);
    const uint32_t metadata_cols = read_value<uint32_t>(in);
    if (metadata_cols != 1 || metadata_rows < 8)
        throw std::runtime_error("Disk index metadata header is not in expected bin format");

    DiskMetadata meta;
    meta.nodes = read_value<uint64_t>(in);
    meta.dims = read_value<uint64_t>(in);
    meta.medoid = read_value<uint64_t>(in);
    meta.max_node_len = read_value<uint64_t>(in);
    meta.nodes_per_sector = read_value<uint64_t>(in);
    meta.frozen_points = read_value<uint64_t>(in);
    meta.frozen_location = read_value<uint64_t>(in);
    meta.reorder_exists = read_value<uint64_t>(in);

    if (meta.nodes == 0 || meta.dims == 0 || meta.max_node_len == 0)
        throw std::runtime_error("Disk index metadata contains invalid zero values");
    return meta;
}

bool validate_disk_node_layout(std::ifstream &in, const DiskMetadata &meta, uint64_t coord_bytes)
{
    if (coord_bytes + sizeof(uint32_t) > meta.max_node_len)
        return false;

    const uint64_t max_degree = (meta.max_node_len - coord_bytes) / sizeof(uint32_t) - 1;
    const uint64_t num_sectors_per_node = meta.nodes_per_sector > 0 ? 1 : div_round_up(meta.max_node_len, SECTOR_LEN);
    std::vector<char> buf(static_cast<size_t>(num_sectors_per_node * SECTOR_LEN));

    const uint64_t nodes_to_check = std::min<uint64_t>(meta.nodes, 1024);
    for (uint64_t node = 0; node < nodes_to_check; ++node)
    {
        const uint64_t sector = 1 + (meta.nodes_per_sector > 0 ? node / meta.nodes_per_sector
                                                               : node * div_round_up(meta.max_node_len, SECTOR_LEN));
        const uint64_t offset_in_sector = meta.nodes_per_sector > 0 ? (node % meta.nodes_per_sector) * meta.max_node_len : 0;

        in.seekg(static_cast<std::streamoff>(sector * SECTOR_LEN), std::ios::beg);
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        if (!in)
            return false;

        const char *node_buf = buf.data() + offset_in_sector;
        uint32_t degree = 0;
        std::memcpy(&degree, node_buf + coord_bytes, sizeof(uint32_t));
        if (degree == 0 || degree > max_degree)
            return false;

        const uint32_t *neighbors = reinterpret_cast<const uint32_t *>(node_buf + coord_bytes + sizeof(uint32_t));
        const uint32_t check_degree = std::min<uint32_t>(degree, 32);
        for (uint32_t i = 0; i < check_degree; ++i)
        {
            if (neighbors[i] >= meta.nodes)
                return false;
        }
    }
    return true;
}

uint64_t choose_coord_bytes(const std::string &path, const DiskMetadata &meta, const std::string &data_type)
{
    if (data_type == "float")
        return meta.dims * sizeof(float);
    if (data_type == "int8" || data_type == "uint8")
        return meta.dims;
    if (data_type != "auto")
        throw std::runtime_error("Unknown data_type: " + data_type);

    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Could not open disk index file " + path);

    const uint64_t one_byte = meta.dims;
    const uint64_t four_byte = meta.dims * sizeof(float);
    const bool one_ok = validate_disk_node_layout(in, meta, one_byte);
    in.clear();
    const bool four_ok = validate_disk_node_layout(in, meta, four_byte);

    if (four_ok && !one_ok)
        return four_byte;
    if (one_ok && !four_ok)
        return one_byte;
    if (one_ok)
        return one_byte;

    throw std::runtime_error("Could not infer disk node coordinate size. Retry with --data_type float|int8|uint8.");
}

void export_disk_graph(const std::string &path, const std::string &output_csv, const std::string &data_type)
{
    const DiskMetadata meta = read_disk_metadata(path);
    const uint64_t coord_bytes = choose_coord_bytes(path, meta, data_type);
    const uint64_t max_degree = (meta.max_node_len - coord_bytes) / sizeof(uint32_t) - 1;
    const uint64_t num_sectors_per_node = meta.nodes_per_sector > 0 ? 1 : div_round_up(meta.max_node_len, SECTOR_LEN);

    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("Could not open disk index file " + path);

    std::ofstream out(output_csv);
    if (!out)
        throw std::runtime_error("Could not open output CSV " + output_csv);
    out << "node_id,neighbor_id\n";

    std::vector<char> buf(static_cast<size_t>(num_sectors_per_node * SECTOR_LEN));
    std::vector<uint32_t> neighbors;
    for (uint64_t node = 0; node < meta.nodes; ++node)
    {
        const uint64_t sector = 1 + (meta.nodes_per_sector > 0 ? node / meta.nodes_per_sector
                                                               : node * div_round_up(meta.max_node_len, SECTOR_LEN));
        const uint64_t offset_in_sector = meta.nodes_per_sector > 0 ? (node % meta.nodes_per_sector) * meta.max_node_len : 0;

        in.seekg(static_cast<std::streamoff>(sector * SECTOR_LEN), std::ios::beg);
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        if (!in)
            throw std::runtime_error("Unexpected end of file while reading disk node sector");

        const char *node_buf = buf.data() + offset_in_sector;
        uint32_t degree = 0;
        std::memcpy(&degree, node_buf + coord_bytes, sizeof(uint32_t));
        if (degree > max_degree)
            throw std::runtime_error("Found disk node degree larger than inferred max degree");

        neighbors.resize(degree);
        std::memcpy(neighbors.data(), node_buf + coord_bytes + sizeof(uint32_t), sizeof(uint32_t) * uint64_t(degree));
        write_edges(out, static_cast<uint32_t>(node), neighbors.data(), degree);
    }

    std::cerr << "Exported " << meta.nodes << " disk-index nodes from " << path << " to " << output_csv
              << " (medoid=" << meta.medoid << ", coord_bytes_per_node=" << coord_bytes
              << ", frozen_points=" << meta.frozen_points << ")." << std::endl;
}

struct Options
{
    std::string index_prefix;
    std::string output_csv;
    std::string format = "auto";
    std::string data_type = "auto";
};

void print_usage(const char *program)
{
    std::cerr << "Usage: " << program
              << " --index_prefix <prefix-or-index-file> --output <graph.csv> "
                 "[--format auto|vamana|disk] [--data_type auto|float|int8|uint8]\n"
              << "       " << program << " <prefix-or-index-file> <graph.csv>\n";
}

Options parse_args(int argc, char **argv)
{
    Options opts;
    if (argc == 3)
    {
        opts.index_prefix = argv[1];
        opts.output_csv = argv[2];
        return opts;
    }

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto read_arg_value = [&](const std::string &name) {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + name);
            return std::string(argv[++i]);
        };

        if (arg == "--index_prefix" || arg == "--index" || arg == "-i")
            opts.index_prefix = read_arg_value(arg);
        else if (arg == "--output" || arg == "-o")
            opts.output_csv = read_arg_value(arg);
        else if (arg == "--format")
            opts.format = read_arg_value(arg);
        else if (arg == "--data_type")
            opts.data_type = read_arg_value(arg);
        else if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            std::exit(0);
        }
        else
            throw std::runtime_error("Unknown argument: " + arg);
    }

    if (opts.index_prefix.empty() || opts.output_csv.empty())
        throw std::runtime_error("Both --index_prefix and --output are required");
    return opts;
}

std::string resolve_disk_path(const std::string &prefix)
{
    if (file_exists(prefix))
        return prefix;
    if (file_exists(prefix + "_disk.index"))
        return prefix + "_disk.index";
    throw std::runtime_error("Could not find disk index file at " + prefix + " or " + prefix + "_disk.index");
}

std::string resolve_vamana_path(const std::string &prefix)
{
    if (file_exists(prefix))
        return prefix;
    if (file_exists(prefix + "_mem.index"))
        return prefix + "_mem.index";
    throw std::runtime_error("Could not find Vamana graph file at " + prefix + " or " + prefix + "_mem.index");
}

} // namespace

int main(int argc, char **argv)
{
    try
    {
        const Options opts = parse_args(argc, argv);
        if (opts.format == "vamana")
        {
            export_vamana_graph(resolve_vamana_path(opts.index_prefix), opts.output_csv);
        }
        else if (opts.format == "disk")
        {
            export_disk_graph(resolve_disk_path(opts.index_prefix), opts.output_csv, opts.data_type);
        }
        else if (opts.format == "auto")
        {
            if (file_exists(opts.index_prefix) && looks_like_vamana_graph(opts.index_prefix))
                export_vamana_graph(opts.index_prefix, opts.output_csv);
            else if (file_exists(opts.index_prefix + "_mem.index") && looks_like_vamana_graph(opts.index_prefix + "_mem.index"))
                export_vamana_graph(opts.index_prefix + "_mem.index", opts.output_csv);
            else
                export_disk_graph(resolve_disk_path(opts.index_prefix), opts.output_csv, opts.data_type);
        }
        else
        {
            throw std::runtime_error("Unknown format: " + opts.format);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "export_graph_topology failed: " << e.what() << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
