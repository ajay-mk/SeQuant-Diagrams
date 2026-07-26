#include <SeQuant/core/context.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/expressions/product.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
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

static std::string narrow(std::wstring_view w) {
  return std::string(w.begin(), w.end());  // ASCII labels only
}

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

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: sq-diagram-extract \"<DSL term>\"\n";
    return 1;
  }
  set_default_context(
      Context({.index_space_registry_shared_ptr = mbpt::make_min_sr_spaces(),
               .vacuum = Vacuum::SingleProduct}));

  const std::string narrow_in(argv[1]);
  const std::wstring input(narrow_in.begin(), narrow_in.end());
  ExprPtr expr = deserialize<ExprPtr>(input);

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

  std::ostringstream out;
  out << "{\"term\":\"" << json_escape(narrow_in) << "\",";
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
  std::cout << out.str() << std::endl;
  return 0;
}
