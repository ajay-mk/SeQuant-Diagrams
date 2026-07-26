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
  return "fock";  // ponytail: fall back to one-body glyph for unknown labels
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
    prefactor = narrow(to_latex(Constant(prod.scalar())));
    factors.assign(prod.factors().begin(), prod.factors().end());
  } else {
    factors.push_back(expr);
  }

  out << "{\"term\":\"" << json_escape(term) << "\",";
  out << "\"prefactor\":\"" << json_escape(prefactor) << "\",";
  out << "\"vertices\":[";
  for (size_t i = 0; i < factors.size(); ++i) {
    const auto t = std::dynamic_pointer_cast<Tensor>(factors[i]);
    const std::wstring_view label = t->label();
    out << (i ? "," : "") << "{\"id\":" << i << ",\"kind\":\"" << kind_of(label)
        << "\"" << ",\"label\":\"" << narrow(label) << "\""
        << ",\"bra\":" << json_str_array(labels_of(t->bra()))
        << ",\"ket\":" << json_str_array(labels_of(t->ket())) << "}";
  }
  out << "],";

  // lines from the tensor network edges
  TensorNetworkV1 tn(factors);
  const auto isr = get_default_context().index_space_registry();
  out << "\"lines\":[";
  const auto& edges = tn.edges();
  bool first = true;
  for (const auto& e : edges) {
    const Index& idx = e.idx();
    const bool hole = isr->is_pure_occupied(idx.space());
    const bool external = (e.size() == 1);
    out << (first ? "" : ",") << "{\"index\":\"" << narrow(idx.label()) << "\""
        << ",\"type\":\"" << (hole ? "hole" : "particle") << "\""
        << ",\"external\":" << (external ? "true" : "false")
        << ",\"endpoints\":[";
    for (std::size_t k = 0; k < e.size(); ++k) {
      const auto& term = e[k];
      out << (k ? "," : "") << "{\"vertex\":" << term.tensor_ord
          << ",\"slot\":\"" << slot_name(term.slot_type) << "\""
          << ",\"pos\":" << term.slot_group_ord << "}";
    }
    out << "]}";
    first = false;
  }
  out << "]}";
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: sq-diagram-extract \"<DSL term or sum>\"\n";
    return 1;
  }
  set_default_context(
      Context({.index_space_registry_shared_ptr = mbpt::make_min_sr_spaces(),
               .vacuum = Vacuum::SingleProduct}));

  const std::string narrow_in(argv[1]);
  ExprPtr expr = deserialize<ExprPtr>(toUtf16(narrow_in));

  // A Sum becomes a JSON array, one diagram per summand; a single term stays a
  // bare object so the one-term callers keep working.
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
  return 0;
}
