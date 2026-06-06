#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <cctype>
#include <fstream>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/utility.hpp> 

namespace py = pybind11;

// Required to use std::pair as an unordered_map key
struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1,T2> &p) const {
        return std::hash<T1>{}(p.first) ^ (std::hash<T2>{}(p.second) << 1);
    }
};

typedef std::pair<int, int> TokenPair;

struct PQElement {
    int count;
    TokenPair pair;
    bool operator<(const PQElement& other) const {
        return count < other.count; // Max-heap
    }
};

class BPETokenizer {
private:
    std::unordered_map<TokenPair, int, PairHash> merge_ranks;
    std::unordered_map<TokenPair, int, PairHash> pair_to_token;
    std::unordered_map<int, std::string> vocab;
    int next_token = 256;

    // Internal text pre-processor: 
    // Attaches spaces to the beginning of words to allow cross-word boundary merges.
    // Isolates punctuation to prevent characters like '.' from merging into alphabetic stems.
    std::vector<std::string> pre_tokenize(const std::string& text) {
        std::vector<std::string> words;
        std::string current = "";
        for (char c : text) {
            if (c == ' ' || c == '\n') {
                if (!current.empty()) words.push_back(current);
                current = std::string(1, c);
            } else if (std::ispunct(c)) {
                if (!current.empty()) words.push_back(current);
                current = std::string(1, c);
            } else {
                current += c;
            }
        }
        if (!current.empty()) words.push_back(current);
        return words;
    }

    friend class cereal::access;

    template<class Archive>
    void serialize(Archive & archive) {
        // You simply list the variables. Cereal automatically calculates 
        // byte sizes, string lengths, and map iterations.
        archive(next_token, vocab, merge_ranks, pair_to_token);
    }

public:
    BPETokenizer() {
        // Initialize base vocabulary (bytes 0-255)
        for (int i = 0; i < 256; ++i) {
            vocab[i] = std::string(1, static_cast<char>(i));
        }
    }

    void train(const std::string& corpus, int num_merges) {
        // 1. Text Preprocessing & Frequency Mapping
        std::vector<std::string> word_list = pre_tokenize(corpus);
        std::unordered_map<std::string, int> word_counts;
        for (const std::string& w : word_list) {
            word_counts[w]++;
        }

        // State containers for the O(N log N) merge loop
        std::vector<std::vector<int>> word_tokens;
        std::vector<std::vector<int>> word_prev;
        std::vector<std::vector<int>> word_next;
        std::vector<int> word_heads;
        std::vector<int> word_freqs;

        std::unordered_map<TokenPair, int, PairHash> pair_counts;
        std::unordered_map<TokenPair, std::unordered_set<int>, PairHash> pair_to_words;
        std::priority_queue<PQElement> pq;

        // 2. Data Structure Initialization
        for (const auto& kv : word_counts) {
            const std::string& word = kv.first;
            int freq = kv.second;
            int w = word_tokens.size();
            
            if (word.empty()) continue;

            std::vector<int> tokens(word.size());
            std::vector<int> prev(word.size());
            std::vector<int> next(word.size());
            
            for (size_t i = 0; i < word.size(); ++i) {
                tokens[i] = static_cast<unsigned char>(word[i]);
                prev[i] = static_cast<int>(i) - 1; 
                next[i] = (i + 1 == word.size()) ? -1 : static_cast<int>(i) + 1;
            }
            
            word_tokens.push_back(tokens);
            word_prev.push_back(prev);
            word_next.push_back(next);
            word_freqs.push_back(freq);
            word_heads.push_back(0); 
            
            for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                TokenPair p = {tokens[i], tokens[i+1]};
                pair_counts[p] += freq;
                pair_to_words[p].insert(w);
            }
        }

        for (const auto& kv : pair_counts) {
            pq.push({kv.second, kv.first});
        }

        // 3. The Merge Loop
        for (int i = 0; i < num_merges; ++i) {
            TokenPair best_pair;
            int max_freq = 0;

            // Lazy Deletion
            while (!pq.empty()) {
                auto top = pq.top();
                pq.pop();
                if (top.count == pair_counts[top.pair]) { 
                    best_pair = top.pair;
                    max_freq = top.count;
                    break;
                }
            }

            if (max_freq < 1) break; 

            int new_token = next_token++;
            
            // Record internal state for encoding/decoding later
            merge_ranks[best_pair] = i; 
            vocab[new_token] = vocab[best_pair.first] + vocab[best_pair.second];
            pair_to_token[best_pair] = new_token;

            auto affected_words = pair_to_words[best_pair];
            pair_counts[best_pair] = 0; // Kill stale entries in PQ

            for (int w : affected_words) {
                int freq = word_freqs[w];
                int curr = word_heads[w];
                
                while (curr != -1 && word_next[w][curr] != -1) {
                    int nxt = word_next[w][curr];
                    
                    if (word_tokens[w][curr] == best_pair.first && word_tokens[w][nxt] == best_pair.second) {
                        
                        int prv = word_prev[w][curr];
                        if (prv != -1) {
                            TokenPair left_pair = {word_tokens[w][prv], word_tokens[w][curr]};
                            pair_counts[left_pair] -= freq;
                            
                            TokenPair new_left_pair = {word_tokens[w][prv], new_token};
                            pair_counts[new_left_pair] += freq;
                            pq.push({pair_counts[new_left_pair], new_left_pair});
                            pair_to_words[new_left_pair].insert(w);
                        }
                        
                        int nxt_nxt = word_next[w][nxt];
                        if (nxt_nxt != -1) {
                            TokenPair right_pair = {word_tokens[w][nxt], word_tokens[w][nxt_nxt]};
                            pair_counts[right_pair] -= freq;
                            
                            TokenPair new_right_pair = {new_token, word_tokens[w][nxt_nxt]};
                            pair_counts[new_right_pair] += freq;
                            pq.push({pair_counts[new_right_pair], new_right_pair});
                            pair_to_words[new_right_pair].insert(w);
                        }
                        
                        // O(1) Pointer Swap
                        word_tokens[w][curr] = new_token;
                        word_next[w][curr] = nxt_nxt;
                        if (nxt_nxt != -1) {
                            word_prev[w][nxt_nxt] = curr;
                        }
                        
                        curr = nxt_nxt; 
                    } else {
                        curr = nxt;
                    }
                }
            }
        }
    }

    // Encodes arbitrary text into integer token IDs based on learned merges
    std::vector<int> encode(const std::string& text) {
        std::vector<std::string> words = pre_tokenize(text);
        std::vector<int> encoded_tokens;

        for (const std::string& word : words) {
            std::vector<int> tokens;
            for (char c : word) tokens.push_back(static_cast<unsigned char>(c));

            // Greedily apply merges in the exact order they were learned
            while (tokens.size() > 1) {
                int min_rank = -1;
                size_t best_idx = -1;
                TokenPair best_pair;

                for (size_t i = 0; i < tokens.size() - 1; ++i) {
                    TokenPair p = {tokens[i], tokens[i+1]};
                    auto it = merge_ranks.find(p);
                    if (it != merge_ranks.end()) {
                        if (min_rank == -1 || it->second < min_rank) {
                            min_rank = it->second;
                            best_idx = i;
                            best_pair = p;
                        }
                    }
                }

                if (min_rank == -1) break; // No more eligible merges in this word

                // Apply the best merge
                int target_token = -1;
                auto it_p = pair_to_token.find(best_pair);
                if (it_p != pair_to_token.end()) {
                    target_token = it_p->second;
                }

                std::vector<int> merged_tokens;
                for (size_t i = 0; i < tokens.size(); ++i) {
                    if (i == best_idx) {
                        merged_tokens.push_back(target_token);
                        ++i; // Skip the absorbed token
                    } else {
                        merged_tokens.push_back(tokens[i]);
                    }
                }
                tokens = merged_tokens;
            }
            
            // Append word tokens to the final sequence
            encoded_tokens.insert(encoded_tokens.end(), tokens.begin(), tokens.end());
        }
        return encoded_tokens;
    }

    // Decodes integer tokens back to the exact string representation
    std::string decode(const std::vector<int>& tokens) {
        std::string result = "";
        for (int t : tokens) {
            if (vocab.find(t) != vocab.end()) {
                result += vocab[t];
            } else {
                result += "?"; // Fallback for out-of-bounds IDs
            }
        }
        return result;
    }

    int get_vocab_size() const {
        return vocab.size();
    }

    std::unordered_map<int, std::string> get_vocab() const {
        return vocab;
    }


    void save(const std::string& filepath) {
        std::ofstream os(filepath, std::ios::binary);
        if (!os) throw std::runtime_error("Failed to open file for saving tokenizer.");
        
        cereal::BinaryOutputArchive archive(os);
        archive(*this); 
    }

    void load(const std::string& filepath) {
        std::ifstream is(filepath, std::ios::binary);
        if (!is) throw std::runtime_error("Failed to open file for loading tokenizer.");
        
        // Clearing state is still required before overwriting
        vocab.clear();
        merge_ranks.clear();
        pair_to_token.clear();
        
        cereal::BinaryInputArchive archive(is);
        archive(*this);
    }

    
};

PYBIND11_MODULE(bpe_tokenizer, m) {
    py::class_<BPETokenizer>(m, "BPETokenizer")
        .def(py::init<>())
        .def("train", &BPETokenizer::train, py::arg("corpus"), py::arg("num_merges"))
        .def("encode", &BPETokenizer::encode, py::arg("text"))
        .def("decode", &BPETokenizer::decode, py::arg("tokens"))
        .def("get_vocab_size", &BPETokenizer::get_vocab_size)
        .def("get_vocab", &BPETokenizer::get_vocab)
        .def("save", &BPETokenizer::save, py::arg("filepath"))
        .def("load", &BPETokenizer::load, py::arg("filepath"));
}