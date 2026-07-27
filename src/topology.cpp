#include <SeQuant/core/context.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/expressions/product.hpp>
#include <SeQuant/core/expressions/sum.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
#include <SeQuant/core/utility/string.hpp>
#include <SeQuant/core/index.hpp>
#include <SeQuant/core/index_space_registry.hpp>
#include <SeQuant/core/io/shorthands.hpp>
#include <SeQuant/core/tensor_network/slot.hpp>
#include <SeQuant/core/tensor_network/v1.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>

#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace sequant;

// SeQuant's own UTF-8 codec: labels like t⁺ are not ASCII, so a range copy
// would mangle them in both directions.
static std::string narrow(std::wstring_view w) { return toUtf8(w); }

/// escape the two characters that would break a JSON string literal
static std::string json_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    if (c == '\\' || c == '"') out += '\\';
    out += c;
  }
  return out;
}

static std::string kind_of(std::wstring_view label) {
  if (label == L"f") return "fock";
  if (label == L"g") return "eri";
  if (label == L"t") return "ampl";
  if (label == L"t⁺") return "deexc";  // t⁺, the de-excitation amplitude
  // Falling back to a one-body glyph drew a rank-2 operator as a Fock cross and
  // interpreted it as f_{ijab} -- confidently wrong physics, silently. There is
  // no sane default here: the caller has to say what the operator is.
  throw std::runtime_error("unknown tensor label '" + narrow(label) +
                           "'; known labels are f, g, t, t⁺, Â");
}

static std::string json_str_array(const std::vector<std::string>& xs) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < xs.size(); ++i)
    os << (i ? "," : "") << "\"" << xs[i] << "\"";
  os << "]";
  return os.str();
}

static std::string slot_name(TensorIndexSlotType s) {
  switch (s) {
    case TensorIndexSlotType::Bra:
      return "bra";
    case TensorIndexSlotType::Ket:
      return "ket";
    default:
      return "aux";
  }
}

static std::vector<std::string> labels_of(const auto& indices) {
  std::vector<std::string> out;
  for (const auto& idx : indices) out.push_back(narrow(idx.label()));
  return out;
}

static void emit_diagram(std::ostringstream& out, const ExprPtr& expr,
                         const std::string& term) {
  // A bare single tensor deserializes to Tensor, not Product; normalize both to
  // (prefactor string, factor list).
  std::string prefactor = "1";
  std::vector<ExprPtr> factors;
  if (expr->is<Product>()) {
    const auto& prod = expr->as<Product>();
    // the DSL form ("1/4", "-1/2"), not to_latex: the LaTeX output carries a
    // brace nesting depth that the consumer would have to string-match, which
    // is the same coupling to SeQuant internals that slot_group_ord was
    prefactor = narrow(serialize(ex<Constant>(prod.scalar())));
    factors.assign(prod.factors().begin(), prod.factors().end());
  } else {
    factors.push_back(expr);
  }

  // Â is the projection onto the target manifold, not a vertex. Keeping it out
  // of the tensor network leaves its indices appearing once, so they fall out
  // as external lines -- which is exactly what target indices are (rule 1).
  std::vector<ExprPtr> network;
  std::vector<std::string> target_bra, target_ket;
  for (const auto& f : factors) {
    const auto t = std::dynamic_pointer_cast<Tensor>(f);
    if (t && t->label() == L"Â") {
      target_bra = labels_of(t->bra());
      target_ket = labels_of(t->ket());
    } else {
      network.push_back(f);
    }
  }

  out << "{\"term\":\"" << json_escape(term) << "\",";
  out << "\"prefactor\":\"" << json_escape(prefactor) << "\",";
  // bra/ket kept apart: slot k of each pair up, which is how rule 8 closes
  // external lines into quasiloops
  out << "\"targets\":{\"bra\":" << json_str_array(target_bra)
      << ",\"ket\":" << json_str_array(target_ket) << "},";
  out << "\"vertices\":[";
  std::vector<std::vector<std::string>> bras, kets;
  for (size_t i = 0; i < network.size(); ++i) {
    const auto t = std::dynamic_pointer_cast<Tensor>(network[i]);
    if (!t)
      throw std::runtime_error("factor " + std::to_string(i) +
                               " is not a tensor; only products of tensors "
                               "become diagrams");
    const std::wstring_view label = t->label();
    bras.push_back(labels_of(t->bra()));
    kets.push_back(labels_of(t->ket()));
    out << (i ? "," : "") << "{\"id\":" << i << ",\"kind\":\"" << kind_of(label)
        << "\"" << ",\"label\":\"" << narrow(label) << "\""
        << ",\"bra\":" << json_str_array(bras.back())
        << ",\"ket\":" << json_str_array(kets.back()) << "}";
  }
  out << "],";

  // Terminal::slot_group_ord is identically 0 for symmetric/antisymmetric
  // tensors (v1.hpp; v1.cpp:867 only increments it when nonsymmetric), so it
  // cannot say *which* bra/ket slot a terminal occupies. Recover that from the
  // index's position in the tensor's own slot list instead.
  const auto slot_pos = [&](int tensor_ord, TensorIndexSlotType st,
                            const std::string& label) -> std::size_t {
    const auto& slots = (st == TensorIndexSlotType::Bra) ? bras[tensor_ord]
                                                         : kets[tensor_ord];
    for (std::size_t k = 0; k < slots.size(); ++k)
      if (slots[k] == label) return k;
    // This function exists because slot_group_ord silently returned 0; it must
    // not repeat the trick. A miss would collapse loops and flip signs unseen.
    throw std::logic_error("index '" + label +
                           "' is not among the slots of tensor " +
                           std::to_string(tensor_ord));
  };

  // lines from the tensor network edges
  TensorNetworkV1 tn(network);
  const auto isr = get_default_context().index_space_registry();
  out << "\"lines\":[";
  const auto& edges = tn.edges();
  bool first = true;
  for (const auto& e : edges) {
    const Index& idx = e.idx();
    const bool hole = isr->is_pure_occupied(idx.space());
    // "hole else particle" would draw a general-space index as an upward
    // particle line without saying so; a diagram has no glyph for one.
    if (!hole && !isr->is_pure_unoccupied(idx.space()))
      throw std::runtime_error(
          "index '" + narrow(idx.label()) +
          "' is neither pure-occupied nor pure-unoccupied; diagrams need a "
          "hole/particle split");
    const bool external = (e.size() == 1);
    out << (first ? "" : ",") << "{\"index\":\"" << narrow(idx.label()) << "\""
        << ",\"type\":\"" << (hole ? "hole" : "particle") << "\""
        << ",\"external\":" << (external ? "true" : "false")
        << ",\"endpoints\":[";
    const std::string idx_label = narrow(idx.label());
    for (std::size_t k = 0; k < e.size(); ++k) {
      const auto& term = e[k];
      out << (k ? "," : "") << "{\"vertex\":" << term.tensor_ord
          << ",\"slot\":\"" << slot_name(term.slot_type) << "\""
          << ",\"pos\":" << slot_pos(term.tensor_ord, term.slot_type, idx_label)
          << "}";
    }
    out << "]}";
    first = false;
  }
  out << "]}";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: sq-diagram-topology \"<DSL term or sum>\"\n";
    return 1;
  }
  set_default_context(
      Context({.index_space_registry_shared_ptr = mbpt::make_min_sr_spaces(),
               .vacuum = Vacuum::SingleProduct}));

  const std::string narrow_in(argv[1]);
  try {
    ExprPtr expr = deserialize<ExprPtr>(toUtf16(narrow_in));
    if (!expr) throw std::runtime_error("empty expression");

    // A Sum becomes a JSON array, one diagram per summand; a single term stays
    // a bare object so the one-term callers keep working.
    std::ostringstream out;
    if (expr->is<Sum>()) {
      out << "[";
      bool first = true;
      for (const auto& summand : expr->as<Sum>().summands()) {
        if (!first) out << ",";
        emit_diagram(out, summand, narrow(serialize(summand)));
        first = false;
      }
      out << "]";
    } else {
      emit_diagram(out, expr, narrow_in);
    }
    std::cout << out.str() << std::endl;
  } catch (const std::exception& e) {
    // A DSL typo used to reach a null deref and take the process out with
    // SIGSEGV, which says nothing about what was wrong with the input.
    std::cerr << "sq-diagram-topology: " << e.what() << "\n  input: " << narrow_in
              << "\n";
    return 1;
  }
  return 0;
}
