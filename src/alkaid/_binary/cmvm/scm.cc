#include "scm.hh"
#include "state_opr.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

    // Signed-digit coefficients indexed by shift; pos/neg must never overlap.
    struct Row {
        uint64_t pos = 0;
        uint64_t neg = 0;

        bool empty() const { return (pos | neg) == 0; }
        int popcount() const { return __builtin_popcountll(pos) + __builtin_popcountll(neg); }
        int8_t sign_at(int s) const {
            if ((pos >> s) & 1u)
                return +1;
            if ((neg >> s) & 1u)
                return -1;
            return 0;
        }
        bool has(int s) const { return ((pos | neg) >> s) & 1u; }
        void clear_at(int s) {
            pos &= ~(1ULL << s);
            neg &= ~(1ULL << s);
        }
        void set(int s, int8_t sign) {
            clear_at(s);
            if (sign > 0)
                pos |= (1ULL << s);
            else if (sign < 0)
                neg |= (1ULL << s);
        }
        bool add(int s, int delta) {
            while (delta != 0) {
                if (s >= 63)
                    return false;
                int8_t cur = sign_at(s);
                int v = cur + delta;
                if (v == 0) {
                    clear_at(s);
                    return true;
                }
                if (v == +1 || v == -1) {
                    set(s, (int8_t)v);
                    return true;
                }
                clear_at(s);
                delta = (v > 0) ? +1 : -1;
                s += 1;
            }
            return true;
        }
    };

    struct VarDef {
        int32_t lhs;
        int32_t rhs;
        int8_t gap;
        int8_t polarity;
    };

    struct Pattern {
        int32_t lhs;
        int32_t rhs;
        int8_t gap;
        int8_t polarity;
        bool operator==(const Pattern &o) const = default;
    };
    struct PatternHash {
        size_t operator()(const Pattern &p) const noexcept {
            return ((uint64_t)(uint32_t)p.lhs << 32) ^ ((uint64_t)(uint32_t)p.rhs << 16) ^
                   ((uint64_t)(uint8_t)p.gap << 8) ^ (uint64_t)(uint8_t)p.polarity;
        }
    };

    struct Expr {
        std::vector<Row> rows;
        std::vector<VarDef> defs;

        int total_popcount() const {
            int c = 0;
            for (const auto &r : rows)
                c += r.popcount();
            return c;
        }
    };

    std::vector<int8_t> csd(int64_t n) {
        std::vector<int8_t> d;
        while (n != 0) {
            if ((n & 1) == 0) {
                d.push_back(0);
                n >>= 1;
            }
            else {
                int8_t u;
                if (n == 1 || n == -1)
                    u = (int8_t)n;
                else {
                    int rem = (int)((n & 3) + 4) & 3;
                    u = (int8_t)(2 - rem);
                }
                d.push_back(u);
                n -= u;
                n >>= 1;
            }
        }
        return d;
    }

    int popcount_signed_digits(const std::vector<int8_t> &d) {
        int c = 0;
        for (auto v : d)
            c += (v != 0);
        return c;
    }

    class SDGenerator {
      public:
        explicit SDGenerator(int max_weight) : max_weight_(max_weight) {}

        const std::vector<std::vector<int8_t>> &all(int64_t n) {
            results_.clear();
            std::vector<int8_t> buf;
            recurse(n, max_weight_, buf);
            return results_;
        }

      private:
        int max_weight_;
        std::vector<std::vector<int8_t>> results_;

        void recurse(int64_t n, int rem, std::vector<int8_t> &buf) {
            if (n == 0) {
                results_.push_back(buf);
                return;
            }
            for (int d : {-1, 0, +1}) {
                if (rem == 0 && d != 0)
                    continue;
                int64_t next = n - d;
                if (next & 1)
                    continue;
                buf.push_back((int8_t)d);
                recurse(next >> 1, rem - (d != 0 ? 1 : 0), buf);
                buf.pop_back();
            }
        }
    };

    struct DigitTriple {
        int n;
        std::array<int, 3> off;
        std::array<int8_t, 3> sign;
    };

    std::vector<std::vector<int8_t>> all_sd_lt3(int64_t m) {
        if (m == 0)
            return {};
        bool neg = m < 0;
        int64_t absm = neg ? -m : m;
        SDGenerator gen(3);
        auto reps = gen.all(absm);
        if (!neg)
            return reps;
        for (auto &r : reps)
            for (auto &d : r)
                d = (int8_t)-d;
        return reps;
    }

    bool to_triple(const std::vector<int8_t> &rep, DigitTriple &out) {
        int n = 0;
        for (int i = 0; i < (int)rep.size(); ++i) {
            if (rep[i] != 0) {
                if (n == 3)
                    return false;
                out.off[n] = i;
                out.sign[n] = rep[i];
                ++n;
            }
        }
        out.n = n;
        return true;
    }

    DigitTriple syntactic_pattern_shape(int gap_i, int8_t sigma) {
        DigitTriple t{};
        t.n = 2;
        t.off = {0, gap_i, 0};
        t.sign = {sigma, +1, 0};
        if (t.off[0] > t.off[1]) {
            std::swap(t.off[0], t.off[1]);
            std::swap(t.sign[0], t.sign[1]);
        }
        return t;
    }

    bool same_triple(const DigitTriple &a, const DigitTriple &b) {
        if (a.n != b.n)
            return false;
        for (int i = 0; i < a.n; ++i) {
            if (a.off[i] != b.off[i] || a.sign[i] != b.sign[i])
                return false;
        }
        return true;
    }

    DigitTriple sort_triple(DigitTriple t) {
        for (int i = 0; i < t.n; ++i)
            for (int j = i + 1; j < t.n; ++j)
                if (t.off[i] > t.off[j]) {
                    std::swap(t.off[i], t.off[j]);
                    std::swap(t.sign[i], t.sign[j]);
                }
        return t;
    }

    struct CompiledTemplate {
        DigitTriple source;
        int gap_i;
        int8_t sigma;
        int q_shift;
        int8_t tau;
        int weight;
    };

    std::vector<CompiledTemplate> build_templates(int max_gap) {
        std::vector<CompiledTemplate> out;
        for (int sigma_i : {+1, -1}) {
            int8_t sigma = (int8_t)sigma_i;
            for (int gap_i = 1; gap_i <= max_gap; ++gap_i) {
                int64_t P_val = (1LL << gap_i) + sigma;
                DigitTriple P_shape = syntactic_pattern_shape(gap_i, sigma);
                for (int tau_i : {+1, -1}) {
                    int8_t tau = (int8_t)tau_i;
                    for (int q = 1; q <= max_gap + 2; ++q) {
                        int64_t mult = (1LL << q) + tau;
                        if (mult == 0)
                            continue;
                        int64_t total = P_val * mult;
                        if (total == 0)
                            continue;
                        auto reps = all_sd_lt3(total);
                        for (const auto &r : reps) {
                            DigitTriple t{};
                            if (!to_triple(r, t))
                                continue;
                            if (t.n < 2)
                                continue;
                            if (t.n == 2) {
                                DigitTriple ts = sort_triple(t);
                                DigitTriple ps = sort_triple(P_shape);
                                if (same_triple(ts, ps))
                                    continue;
                                DigitTriple ts2 = ts;
                                for (int k = 0; k < ts2.n; ++k)
                                    ts2.sign[k] = (int8_t)-ts2.sign[k];
                                if (same_triple(ts2, ps))
                                    continue;
                            }
                            out.push_back({sort_triple(t), gap_i, sigma, q, tau, t.n});
                        }
                    }
                }
            }
        }
        return out;
    }

    // For sigma=-1 the stored variable is oriented opposite to the matched pair.
    inline int8_t to_b_sign(int8_t occ_sign, int8_t polarity) {
        return (int8_t)((polarity > 0) ? occ_sign : (int8_t)-occ_sign);
    }

    inline int row_max_shift(const Row &r) {
        uint64_t m = r.pos | r.neg;
        if (m == 0)
            return -1;
        return 63 - __builtin_clzll(m);
    }
    inline int row_min_shift(const Row &r) {
        uint64_t m = r.pos | r.neg;
        if (m == 0)
            return -1;
        return __builtin_ctzll(m);
    }

    struct UsedSet {
        std::vector<uint64_t> used;
        void ensure(int v) {
            if ((int)used.size() <= v)
                used.resize(v + 1, 0);
        }
        bool any(int v, uint64_t mask) const {
            if (v >= (int)used.size())
                return false;
            return (used[v] & mask) != 0;
        }
        void mark(int v, uint64_t mask) {
            ensure(v);
            used[v] |= mask;
        }
    };

    struct Occurrence {
        int8_t kind;
        int anchor;
        uint64_t source_mask;
        int8_t use_count;
        std::array<int32_t, 2> use_var;
        std::array<uint64_t, 2> use_mask;
        int8_t rep_count;
        std::array<int, 2> rep_shift;
        std::array<int8_t, 2> rep_sign;
    };

    inline void set_single_use(Occurrence &o, int src_var, uint64_t mask) {
        o.source_mask = mask;
        o.use_count = 1;
        o.use_var = {src_var, -1};
        o.use_mask = {mask, 0};
    }

    inline void set_dual_use(Occurrence &o, int lhs_var, uint64_t lhs_mask, int rhs_var, uint64_t rhs_mask) {
        o.source_mask = lhs_mask | rhs_mask;
        o.use_count = 2;
        o.use_var = {lhs_var, rhs_var};
        o.use_mask = {lhs_mask, rhs_mask};
    }

    struct Candidate {
        Pattern pat;
        std::vector<Occurrence> occs;
    };

    inline void
    push_regular_occurrence(std::vector<Occurrence> &out, int src_var, int anchor, int gap_i, int8_t rep_sign) {
        Occurrence o{};
        o.kind = 0;
        o.anchor = anchor;
        set_single_use(o, src_var, (1ULL << anchor) | (1ULL << (anchor + gap_i)));
        o.rep_count = 1;
        o.rep_shift[0] = anchor;
        o.rep_sign[0] = rep_sign;
        out.push_back(o);
    }

    void enumerate_regular(int src_var, const Row &r, int gap_i, int8_t sigma, std::vector<Occurrence> &out) {
        uint64_t pos = r.pos, neg = r.neg;
        if (sigma > 0) {
            uint64_t pp = pos & (pos >> gap_i);
            uint64_t nn = neg & (neg >> gap_i);
            while (pp) {
                int a = __builtin_ctzll(pp);
                pp &= pp - 1;
                push_regular_occurrence(out, src_var, a, gap_i, +1);
            }
            while (nn) {
                int a = __builtin_ctzll(nn);
                nn &= nn - 1;
                push_regular_occurrence(out, src_var, a, gap_i, -1);
            }
        }
        else {
            uint64_t np = neg & (pos >> gap_i);
            uint64_t pn = pos & (neg >> gap_i);
            while (np) {
                int a = __builtin_ctzll(np);
                np &= np - 1;
                push_regular_occurrence(out, src_var, a, gap_i, +1);
            }
            while (pn) {
                int a = __builtin_ctzll(pn);
                pn &= pn - 1;
                push_regular_occurrence(out, src_var, a, gap_i, -1);
            }
        }
    }

    void enumerate_mixed_regular(
        int lhs_var,
        const Row &lhs,
        int rhs_var,
        const Row &rhs,
        int gap_i,
        int8_t sigma,
        std::vector<Occurrence> &out
    ) {
        int max_lhs = row_max_shift(lhs);
        int max_rhs = row_max_shift(rhs);
        if (max_lhs < 0 || max_rhs < gap_i)
            return;
        int max_anchor = std::min(max_lhs, max_rhs - gap_i);
        for (int a = 0; a <= max_anchor; ++a) {
            int8_t lhs_sign = lhs.sign_at(a);
            if (lhs_sign == 0)
                continue;
            int8_t rhs_sign = rhs.sign_at(a + gap_i);
            if (rhs_sign == 0 || rhs_sign != lhs_sign * sigma)
                continue;
            Occurrence o{};
            o.kind = 0;
            o.anchor = a;
            set_dual_use(o, lhs_var, 1ULL << a, rhs_var, 1ULL << (a + gap_i));
            o.rep_count = 1;
            o.rep_shift[0] = a;
            o.rep_sign[0] = lhs_sign;
            out.push_back(o);
        }
    }

    bool template_matches_at(const Row &r, int anchor, const DigitTriple &shape, int8_t occ_sign) {
        for (int k = 0; k < shape.n; ++k) {
            int s = anchor + shape.off[k];
            if (s >= 64)
                return false;
            int8_t need = (int8_t)(occ_sign * shape.sign[k]);
            if (r.sign_at(s) != need)
                return false;
        }
        return true;
    }

    void enumerate_template(
        int src_var,
        const Row &r,
        const CompiledTemplate &tmpl,
        int8_t kind,
        std::vector<Occurrence> &out
    ) {
        const DigitTriple &shape = tmpl.source;
        int max_off = 0;
        for (int k = 0; k < shape.n; ++k)
            max_off = std::max(max_off, shape.off[k]);
        int max_anchor = 63 - max_off;
        if (max_anchor < 0)
            return;

        uint64_t mask_all = r.pos | r.neg;
        if (mask_all == 0)
            return;
        int max_a = std::min<int>(max_anchor, row_max_shift(r));

        for (int a = 0; a <= max_a; ++a) {
            for (int8_t occ : {+1, -1}) {
                if (!template_matches_at(r, a, shape, occ))
                    continue;
                Occurrence o{};
                o.kind = kind;
                o.anchor = a;
                uint64_t m = 0;
                for (int k = 0; k < shape.n; ++k)
                    m |= 1ULL << (a + shape.off[k]);
                set_single_use(o, src_var, m);
                // shape == ((1<<q) + tau) * P.
                o.rep_count = 2;
                o.rep_shift[0] = a;
                o.rep_sign[0] = (int8_t)(occ * tmpl.tau);
                o.rep_shift[1] = a + tmpl.q_shift;
                o.rep_sign[1] = (int8_t)(occ * 1);
                if (o.rep_shift[1] >= 64)
                    continue;
                out.push_back(o);
            }
        }
    }

    struct SelectionResult {
        std::vector<Occurrence> normal_picks;
        std::vector<Occurrence> pseudo_picks;
        int score;
    };

    inline bool occurrence_uses_boundary(const Occurrence &o, int boundary) {
        return o.kind == 1 && boundary >= 0 && ((o.source_mask >> boundary) & 1ULL) != 0;
    }

    void sort_occurrences(std::vector<const Occurrence *> &occs, int boundary, bool ltr) {
        std::sort(occs.begin(), occs.end(), [&](const Occurrence *a, const Occurrence *b) {
            bool ba = occurrence_uses_boundary(*a, boundary);
            bool bb = occurrence_uses_boundary(*b, boundary);
            if (ba != bb)
                return ba;
            if (a->anchor != b->anchor)
                return ltr ? a->anchor > b->anchor : a->anchor < b->anchor;
            return a->source_mask < b->source_mask;
        });
    }

    bool used_by(const UsedSet &consumed, const Occurrence &o) {
        for (int i = 0; i < o.use_count; ++i)
            if (consumed.any(o.use_var[i], o.use_mask[i]))
                return true;
        return false;
    }

    bool pick_occurrence(UsedSet &consumed, const Occurrence &o, std::vector<Occurrence> &out) {
        if (used_by(consumed, o))
            return false;
        for (int i = 0; i < o.use_count; ++i)
            consumed.mark(o.use_var[i], o.use_mask[i]);
        out.push_back(o);
        return true;
    }

    SelectionResult select_compatible(const Candidate &cand, const Row &r, bool ltr) {
        SelectionResult res{};
        int boundary = ltr ? row_max_shift(r) : row_min_shift(r);

        std::vector<const Occurrence *> regular, odp, pseudo;
        for (const auto &o : cand.occs) {
            if (o.kind == 0)
                regular.push_back(&o);
            else if (o.kind == 1)
                odp.push_back(&o);
            else
                pseudo.push_back(&o);
        }

        sort_occurrences(regular, boundary, ltr);
        sort_occurrences(odp, boundary, ltr);
        sort_occurrences(pseudo, boundary, ltr);

        UsedSet consumed;

        size_t i = 0;
        for (; i < odp.size(); ++i) {
            const Occurrence *o = odp[i];
            if (!occurrence_uses_boundary(*o, boundary))
                break;
            if (pick_occurrence(consumed, *o, res.normal_picks))
                res.score += 1;
        }
        for (auto *o : regular) {
            if (pick_occurrence(consumed, *o, res.normal_picks))
                res.score += 1;
        }
        for (; i < odp.size(); ++i) {
            if (pick_occurrence(consumed, *odp[i], res.normal_picks))
                res.score += 1;
        }
        for (auto *o : pseudo)
            if (!used_by(consumed, *o))
                res.pseudo_picks.push_back(*o);
        return res;
    }

    bool apply_occurrences(Expr &e, int b_var, int8_t polarity, const std::vector<Occurrence> &occs) {
        Row &dst = e.rows[b_var];
        bool same_source_pattern = e.defs[b_var - 1].lhs == e.defs[b_var - 1].rhs;
        for (const auto &o : occs) {
            for (int u = 0; u < o.use_count; ++u) {
                Row &src = e.rows[o.use_var[u]];
                for (int s = 0; s < 64; ++s) {
                    if ((o.use_mask[u] >> s) & 1ULL && !src.has(s))
                        return false;
                }
            }
            for (int u = 0; u < o.use_count; ++u) {
                Row &src = e.rows[o.use_var[u]];
                src.pos &= ~o.use_mask[u];
                src.neg &= ~o.use_mask[u];
            }
            for (int k = 0; k < o.rep_count; ++k) {
                int8_t bsign = same_source_pattern ? to_b_sign(o.rep_sign[k], polarity) : o.rep_sign[k];
                if (!dst.add(o.rep_shift[k], (int)bsign))
                    return false;
            }
        }
        return true;
    }

    std::vector<std::vector<Occurrence>> pseudo_branches(const std::vector<Occurrence> &pseudo) {
        if (pseudo.empty())
            return {};

        struct Group {
            uint64_t mask;
            std::vector<Occurrence> occs;
        };
        std::vector<Group> groups;
        for (const auto &o : pseudo) {
            auto it = std::find_if(groups.begin(), groups.end(), [&](const Group &g) {
                return g.mask == o.source_mask;
            });
            if (it == groups.end())
                groups.push_back(Group{o.source_mask, {o}});
            else
                it->occs.push_back(o);
        }

        struct Branch {
            uint64_t used;
            std::vector<Occurrence> occs;
        };
        std::vector<Branch> branches{{0, {}}};
        for (const auto &g : groups) {
            std::vector<Branch> next;
            for (const auto &branch : branches) {
                if ((branch.used & g.mask) != 0) {
                    next.push_back(branch);
                    continue;
                }
                for (const auto &o : g.occs) {
                    Branch b = branch;
                    b.used |= g.mask;
                    b.occs.push_back(o);
                    next.push_back(std::move(b));
                }
            }
            branches = std::move(next);
        }

        std::vector<std::vector<Occurrence>> out;
        out.reserve(branches.size());
        for (auto &branch : branches) {
            if (!branch.occs.empty())
                out.push_back(std::move(branch.occs));
        }
        return out;
    }

    int finalize_cost(const Expr &e) {
        int defs = (int)e.defs.size();
        int terms = e.total_popcount();
        int sum = std::max(0, terms - 1);
        return defs + sum;
    }

    // Internal invariant: substitutions must preserve the constant in X-units.
    __int128 expr_value_in_x_units(const Expr &e) {
        std::vector<__int128> val(e.rows.size());
        val[0] = 1;
        for (int v = 1; v < (int)e.rows.size(); ++v) {
            const VarDef &d = e.defs[v - 1];
            __int128 lhs = val[d.lhs];
            __int128 rhs = val[d.rhs] << d.gap;
            if (d.polarity > 0)
                val[v] = lhs + rhs;
            else
                val[v] = lhs - rhs;
        }
        __int128 total = 0;
        for (int v = 0; v < (int)e.rows.size(); ++v) {
            const Row &r = e.rows[v];
            for (int s = 0; s < 64; ++s) {
                int8_t sg = r.sign_at(s);
                if (sg == 0)
                    continue;
                __int128 contribution = ((__int128)sg) * val[v];
                total += contribution << s;
            }
        }
        return total;
    }

    struct SearchResult {
        int adders = std::numeric_limits<int>::max();
        std::vector<VarDef> defs;
        std::vector<Row> rows;
    };

    struct CandPick {
        int score;
        Pattern pat;
        SelectionResult sel;
        int existing_var;

        bool exists() const { return existing_var >= 0; }
    };

    class SearchContext {
      public:
        SearchContext(int64_t target, int max_gap, const std::vector<CompiledTemplate> &templates)
            : target_(target), max_gap_(max_gap), templates_(templates) {}

        bool run(const std::vector<int8_t> &sd, bool ltr, int budget, SearchResult &result) const {
            Expr e;
            Row row;
            for (int i = 0; i < (int)sd.size(); ++i)
                if (sd[i] != 0)
                    row.set(i, sd[i]);
            e.rows.push_back(row);
            return search(std::move(e), ltr, budget, result);
        }

      private:
        using ExistingMap = std::unordered_map<Pattern, int, PatternHash>;

        int64_t target_;
        int max_gap_;
        const std::vector<CompiledTemplate> &templates_;

        ExistingMap existing_vars(const Expr &e) const {
            ExistingMap existing;
            for (int v = 1; v < (int)e.rows.size(); ++v) {
                const VarDef &d = e.defs[v - 1];
                existing[Pattern{d.lhs, d.rhs, d.gap, d.polarity}] = v;
            }
            return existing;
        }

        void append_templates(
            int src_var,
            const Row &r,
            int gap_i,
            int8_t sigma,
            std::vector<Occurrence> &occs
        ) const {
            for (const auto &tmpl : templates_) {
                if (tmpl.gap_i != gap_i || tmpl.sigma != sigma)
                    continue;
                enumerate_template(src_var, r, tmpl, tmpl.weight == 3 ? 1 : 2, occs);
            }
        }

        void push_candidate(
            Candidate &&cand,
            const Row &selection_row,
            bool ltr,
            const ExistingMap &existing,
            std::vector<CandPick> &cands,
            int &best_score
        ) const {
            if (cand.occs.empty())
                return;
            SelectionResult sel = select_compatible(cand, selection_row, ltr);
            if (sel.score == 0)
                return;

            auto found = existing.find(cand.pat);
            int existing_var = found == existing.end() ? -1 : found->second;
            int min_score = existing_var >= 0 ? 1 : 2;
            if (sel.score < min_score)
                return;

            best_score = std::max(best_score, sel.score);
            cands.push_back({sel.score, cand.pat, std::move(sel), existing_var});
        }

        std::vector<CandPick> top_candidates(const Expr &e, bool ltr, int &best_score) const {
            auto existing = existing_vars(e);
            std::vector<CandPick> cands;

            for (int src_var = 0; src_var < (int)e.rows.size(); ++src_var) {
                const Row &r = e.rows[src_var];
                if (r.empty())
                    continue;
                int max_g = std::min(max_gap_, row_max_shift(r) - row_min_shift(r));
                for (int gap_i = 1; gap_i <= max_g; ++gap_i) {
                    for (int8_t sigma : {+1, -1}) {
                        Candidate cand{Pattern{src_var, src_var, (int8_t)gap_i, sigma}, {}};
                        enumerate_regular(src_var, r, gap_i, sigma, cand.occs);
                        append_templates(src_var, r, gap_i, sigma, cand.occs);
                        push_candidate(std::move(cand), r, ltr, existing, cands, best_score);
                    }
                }
            }

            for (int lhs_var = 0; lhs_var < (int)e.rows.size(); ++lhs_var) {
                const Row &lhs = e.rows[lhs_var];
                if (lhs.empty())
                    continue;
                for (int rhs_var = 0; rhs_var < (int)e.rows.size(); ++rhs_var) {
                    if (lhs_var == rhs_var)
                        continue;
                    const Row &rhs = e.rows[rhs_var];
                    if (rhs.empty())
                        continue;
                    int max_g = std::min(max_gap_, row_max_shift(rhs));
                    for (int gap_i = 1; gap_i <= max_g; ++gap_i) {
                        for (int8_t sigma : {+1, -1}) {
                            Candidate cand{Pattern{lhs_var, rhs_var, (int8_t)gap_i, sigma}, {}};
                            enumerate_mixed_regular(lhs_var, lhs, rhs_var, rhs, gap_i, sigma, cand.occs);
                            push_candidate(std::move(cand), lhs, ltr, existing, cands, best_score);
                        }
                    }
                }
            }

            std::vector<CandPick> top;
            top.reserve(cands.size());
            for (auto &cand : cands)
                if (cand.score == best_score)
                    top.push_back(std::move(cand));

            std::sort(top.begin(), top.end(), [](const CandPick &a, const CandPick &b) {
                if (a.exists() != b.exists())
                    return a.exists();
                if (a.pat.gap != b.pat.gap)
                    return a.pat.gap < b.pat.gap;
                if (a.pat.polarity != b.pat.polarity)
                    return a.pat.polarity > b.pat.polarity;
                if (a.pat.lhs != b.pat.lhs)
                    return a.pat.lhs < b.pat.lhs;
                return a.pat.rhs < b.pat.rhs;
            });
            return top;
        }

        bool finish(Expr &&e, int budget, SearchResult &result) const {
            int cost = finalize_cost(e);
            if (cost >= budget)
                return false;
            if (expr_value_in_x_units(e) != (__int128)target_)
                throw std::runtime_error("scm: substitution chain produced wrong value (template bug)");
            result.adders = cost;
            result.defs = std::move(e.defs);
            result.rows = std::move(e.rows);
            return true;
        }

        void update_best(Expr &&e, bool ltr, int budget, SearchResult &best, bool &found) const {
            SearchResult child;
            int child_budget = found ? best.adders : budget;
            if (!search(std::move(e), ltr, child_budget, child))
                return;
            if (!found || child.adders < best.adders) {
                best = std::move(child);
                found = true;
            }
        }

        void branch_candidate(
            const Expr &e,
            const CandPick &cand,
            bool ltr,
            int budget,
            SearchResult &best,
            bool &found
        ) const {
            Expr base = e;
            int b_var = cand.existing_var;
            if (!cand.exists()) {
                base.defs.push_back(VarDef{cand.pat.lhs, cand.pat.rhs, cand.pat.gap, cand.pat.polarity});
                base.rows.emplace_back();
                b_var = (int)base.rows.size() - 1;
            }

            Expr normal = base;
            if (apply_occurrences(normal, b_var, cand.pat.polarity, cand.sel.normal_picks))
                update_best(std::move(normal), ltr, budget, best, found);

            for (const auto &pseudo_set : pseudo_branches(cand.sel.pseudo_picks)) {
                Expr with_pseudo = base;
                if (!apply_occurrences(with_pseudo, b_var, cand.pat.polarity, cand.sel.normal_picks))
                    continue;
                if (!apply_occurrences(with_pseudo, b_var, cand.pat.polarity, pseudo_set))
                    continue;
                update_best(std::move(with_pseudo), ltr, budget, best, found);
            }
        }

        bool search(Expr e, bool ltr, int budget, SearchResult &result) const {
            if ((int)e.defs.size() >= budget)
                return false;

            int best_score = 0;
            auto top = top_candidates(e, ltr, best_score);
            if (best_score == 0)
                return finish(std::move(e), budget, result);

            SearchResult best;
            bool found = false;
            for (const auto &cand : top)
                branch_candidate(e, cand, ltr, budget, best, found);
            if (!found)
                return false;

            result = std::move(best);
            return true;
        }
    };

    CombLogicResult assemble_solution(
        const std::vector<VarDef> &defs,
        const std::vector<Row> &rows,
        int t,
        bool out_neg,
        const QInterval &qint
    ) {
        CombLogicResult result;
        result.shape = {1, 1};
        result.inp_shifts = {0};
        result.adder_size = -1;
        result.carry_size = -1;

        std::vector<Op> ops;
        std::vector<QInterval> var_qint;

        ops.push_back(Op{0, -1, -1, 0, qint, 0.0f, 0.0f});
        var_qint.push_back(qint);

        std::vector<int64_t> var_op_idx;
        var_op_idx.push_back(0);

        for (size_t i = 0; i < defs.size(); ++i) {
            const VarDef &d = defs[i];
            int64_t lhs_idx = var_op_idx[d.lhs];
            int64_t rhs_idx = var_op_idx[d.rhs];
            QInterval lhs_q = var_qint[d.lhs];
            QInterval rhs_q = var_qint[d.rhs];
            int64_t opcode = (d.polarity > 0) ? 0 : 1;
            QInterval qint = qint_add(lhs_q, rhs_q, d.gap, false, d.polarity < 0);
            auto [cost, lat] = cost_add(lhs_q, rhs_q, d.gap);
            float opl = std::max(ops[lhs_idx].latency, ops[rhs_idx].latency) + lat;
            Op op{lhs_idx, rhs_idx, opcode, d.gap, qint, opl, cost};
            ops.push_back(op);
            var_op_idx.push_back((int64_t)ops.size() - 1);
            var_qint.push_back(qint);
        }

        struct HeapEntry {
            float lat;
            int64_t sub;
            int64_t left_align;
            float qmin, qmax, qstep;
            int64_t id;
            int64_t shift;
            auto as_tuple() const { return std::tie(lat, sub, left_align, qmin, qmax, qstep, id, shift); }
            bool operator>(const HeapEntry &o) const { return as_tuple() > o.as_tuple(); }
        };
        std::priority_queue<HeapEntry, std::vector<HeapEntry>, std::greater<HeapEntry>> heap;
        auto push_var_term = [&](int v, int shift, int8_t sign) {
            int64_t id = var_op_idx[v];
            QInterval q = var_qint[v];
            float lat = roundf(ops[id].latency);
            int64_t la = (int64_t)std::log2(std::max(std::abs(q.max + q.step), std::abs(q.min))) + shift;
            heap.push({lat, (int64_t)(sign < 0 ? 1 : 0), la, q.min, q.max, q.step, id, (int64_t)shift});
        };

        for (int v = 0; v < (int)rows.size(); ++v) {
            const Row &r = rows[v];
            if (r.empty())
                continue;
            for (int s = 0; s < 64; ++s) {
                int8_t sg = r.sign_at(s);
                if (sg == 0)
                    continue;
                push_var_term(v, s, sg);
            }
        }

        int64_t final_op_idx;
        int64_t final_shift;
        int64_t final_neg;

        if (heap.empty()) {
            QInterval qz{0.0f, 0.0f, 1.0f};
            Op op{0, -1, 5, 0, qz, 0.0f, 0.0f};
            ops.push_back(op);
            final_op_idx = (int64_t)ops.size() - 1;
            final_shift = 0;
            final_neg = 0;
        }
        else if (heap.size() == 1) {
            auto e0 = heap.top();
            final_op_idx = e0.id;
            final_shift = e0.shift;
            final_neg = e0.sub;
        }
        else {
            while (heap.size() > 1) {
                auto e0 = heap.top();
                heap.pop();
                auto e1 = heap.top();
                heap.pop();
                QInterval qint0{e0.qmin, e0.qmax, e0.qstep};
                QInterval qint1{e1.qmin, e1.qmax, e1.qstep};
                int64_t sub0 = e0.sub, sub1 = e1.sub;
                int64_t id0 = e0.id, id1 = e1.id;
                int64_t shift0 = e0.shift, shift1 = e1.shift;
                float lat0 = e0.lat, lat1 = e1.lat;

                QInterval qint;
                Op op;
                int64_t result_shift;

                if (sub0) {
                    int64_t s = shift0 - shift1;
                    qint = qint_add(qint1, qint0, s, sub1 != 0, sub0 != 0);
                    auto [dcost, dlat] = cost_add(qint1, qint0, s);
                    float lat = std::max(lat0, lat1) + dlat;
                    op = Op{id1, id0, 1 ^ sub1, s, qint, lat, dcost};
                    result_shift = shift1;
                }
                else {
                    int64_t s = shift1 - shift0;
                    qint = qint_add(qint0, qint1, s, sub0 != 0);
                    auto [dcost, dlat] = cost_add(qint0, qint1, s);
                    float lat = std::max(lat0, lat1) + dlat;
                    op = Op{id0, id1, sub1, s, qint, lat, dcost};
                    result_shift = shift0;
                }
                int64_t la =
                    (int64_t)std::log2(std::max(std::abs(qint.max + qint.step), std::abs(qint.min))) +
                    result_shift;
                float lat = roundf(op.latency);
                int64_t new_id = (int64_t)ops.size();
                ops.push_back(op);
                heap.push({lat, sub0 & sub1, la, qint.min, qint.max, qint.step, new_id, result_shift});
            }
            auto fe = heap.top();
            final_op_idx = fe.id;
            final_shift = fe.shift;
            final_neg = fe.sub;
        }

        result.out_idxs = {final_op_idx};
        result.out_shifts = {final_shift + (int64_t)t};
        result.out_negs = {(int64_t)((final_neg != 0) ^ (out_neg ? 1 : 0))};
        result.ops = std::move(ops);
        return result;
    }

    int16_t get_lsb_loc_f64(double x) {
        if (x == 0.0 || std::isnan(x)) {
            return 1023;
        }
        uint64_t bits = std::bit_cast<uint64_t>(x);
        uint16_t exp = static_cast<uint16_t>((bits >> 52) & 0x7FF);
        uint64_t mant = bits & 0xFFFFFFFFFFFFF;
        int mtz = __builtin_ctzll(mant + (1ULL << 52));
        return static_cast<int16_t>(exp + mtz - 1075);
    }

} // namespace

CombLogicResult scm(double constant, int k, const QInterval &qint) {
    if (k < 0)
        throw std::invalid_argument("scm: k must be non-negative");
    if (!std::isfinite(constant))
        throw std::invalid_argument("scm: constant must be finite");

    bool out_neg = constant < 0;
    double absC = std::fabs(constant);
    int t = 0;
    int64_t C_odd = 0;
    if (absC == 0.0) {
        return assemble_solution({}, {}, 0, false, qint);
    }
    int lsb = get_lsb_loc_f64(absC);
    double scaled = absC * std::pow(2.0, -lsb);
    double rounded = std::round(scaled);

    C_odd = (int64_t)rounded;
    if ((C_odd & 1) == 0 && C_odd != 0)
        throw std::runtime_error("scm: internal — C_odd should be odd");
    t = lsb;

    auto digits = csd(C_odd);
    int w0 = popcount_signed_digits(digits);
    int max_w = w0 + k;

    int n_bits = (int)digits.size();
    int max_gap = std::max(2, n_bits + 1);
    auto templates = build_templates(max_gap);

    SDGenerator gen(max_w);
    auto sd_forms = gen.all((int64_t)C_odd);

    std::vector<VarDef> best_defs;
    std::vector<Row> best_rows;

    SearchContext search(C_odd, max_gap, templates);
    int budget = std::numeric_limits<int>::max();
    for (const auto &sd : sd_forms) {
        for (bool ltr : {true, false}) {
            SearchResult result;
            if (search.run(sd, ltr, budget, result)) {
                budget = result.adders;
                best_defs = std::move(result.defs);
                best_rows = std::move(result.rows);
            }
        }
    }

    if (best_rows.empty()) {
        Row row0;
        for (int i = 0; i < (int)digits.size(); ++i)
            if (digits[i] != 0)
                row0.set(i, digits[i]);
        best_rows = {row0};
        best_defs = {};
    }

    return assemble_solution(best_defs, best_rows, t, out_neg, qint);
}
