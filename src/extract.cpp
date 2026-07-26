#include <SeQuant/core/context.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/index_space_registry.hpp>
#include <SeQuant/core/io/shorthands.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>

#include <iostream>
#include <string>

using namespace sequant;

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: sq-diagram-extract \"<DSL term>\"\n";
    return 1;
  }
  set_default_context(
      Context({.index_space_registry_shared_ptr = mbpt::make_min_sr_spaces(),
               .vacuum = Vacuum::SingleProduct}));

  const std::string narrow_in(argv[1]);
  const std::wstring input(narrow_in.begin(), narrow_in.end());  // ASCII DSL

  ExprPtr expr = deserialize<ExprPtr>(input);

  // Task 1 smoke: echo the term back re-serialized.
  std::wcout << serialize(expr) << std::endl;
  return 0;
}
