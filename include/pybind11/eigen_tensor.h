/*
    pybind11/eigen_tensor.h: Conversion of multi-dimensional numpy arrays to Eigen
    tensors.

    Copyright (c) 2020 Simon Pfreundschuh <simon.pfreundschuh@chalmers.se>

    All rights reserved. Use of this source code is governed by a
    BSD-style license that can be found in the LICENSE file.
*/

#pragma once

#include "numpy.h"

#if defined(__INTEL_COMPILER)
#  pragma warning(disable: 1682) // implicit conversion of a 64-bit integral type to a smaller integral type (potential portability problem)
#elif defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wconversion"
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#  ifdef __clang__
//   Eigen generates a bunch of implicit-copy-constructor-is-deprecated warnings with -Wdeprecated
//   under Clang, so disable that warning here:
#    pragma GCC diagnostic ignored "-Wdeprecated"
#  endif
#  if __GNUC__ >= 7
#    pragma GCC diagnostic ignored "-Wint-in-bool-context"
#  endif
#endif

#if defined(_MSC_VER)
#  pragma warning(push)
#  pragma warning(disable: 4127) // warning C4127: Conditional expression is constant
#  pragma warning(disable: 4996) // warning C4996: std::unary_negate is deprecated in C++17
#endif

#include <Eigen/Core>
#include <Eigen/CXX11/Tensor>

NAMESPACE_BEGIN(PYBIND11_NAMESPACE)

// Provide a convenience alias for easier pass-by-ref usage with fully dynamic strides:
using EigenDStride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
using EigenIndex = Eigen::Index;
template <typename MatrixType> using EigenDRef = Eigen::Ref<MatrixType, 0, EigenDStride>;
template <typename MatrixType> using EigenDMap = Eigen::Map<MatrixType, 0, EigenDStride>;

NAMESPACE_BEGIN(detail)

struct is_tensor_impl {
    template <typename Scalar, int NumIndices, int Options, typename IndexType>
        static std::true_type check(Eigen::Tensor<Scalar, NumIndices, Options, IndexType> *);
    static std::false_type check(...);
};

template <typename T>
using is_eigen_tensor = decltype(is_tensor_impl::check((intrinsic_t<T>*)nullptr));


////////////////////////////////////////////////////////////////////////////////
// Tensor conformable
////////////////////////////////////////////////////////////////////////////////

/* Captures numpy/eigen conformability status */

template <bool EigenRowMajor> struct EigenTensorConformable {
    std::vector<size_t> dimensions;
    bool conformable = false;
    EigenIndex dims = 0;

    EigenTensorConformable(bool fits = false) : conformable{fits} {}
    // Matrix type:
    EigenTensorConformable(std::vector<size_t> dimensions_)
            :
    conformable{true}, dimensions(dimensions_) {
    }

    template <typename props> bool stride_compatible() const {
        // To have compatible strides, we need (on both dimensions) one of fully dynamic strides,
        // matching strides, or a dimension size of 1 (in which case the stride value is irrelevant)
        return true;
    }

    operator bool() const { return conformable; }

private:


};

template <size_t N, size_t i = 0>
    struct tensor_dimensions {
        static constexpr auto text = _({static_cast<char>('n' + i)}) + _(", ") + tensor_dimensions<N-1, i + 1>::text;
    };

template <size_t i>
struct tensor_dimensions<0, i> {
    static constexpr auto text = _({static_cast<char>('n' + i)});
};

////////////////////////////////////////////////////////////////////////////////
// EigenProps
////////////////////////////////////////////////////////////////////////////////


// Helper struct for extracting information from an Eigen type
template <typename Type_> struct EigenProps {
    using Type = Type_;
    using Scalar = typename Type::Scalar;
    //using StrideType = typename eigen_extract_stride<Type>::type;
    static constexpr EigenIndex dimensions = Type::NumIndices;
    static constexpr bool row_major = Type::Layout == Eigen::RowMajor;


    // Takes an input array and determines whether we can make it fit into the Eigen type.  If
    // the array is a vector, we attempt to fit it into either an Eigen 1xN or Nx1 vector
    // (preferring the latter if it will fit in either, i.e. for a fully dynamic matrix type).
    static EigenTensorConformable<row_major> conformable(const array &a) {
        const auto dims = a.ndim();
        if (dims < 2) {
            return false;
        }

        std::vector<size_t> dimensions{};
        for (size_t i = 0; i < dims; ++i) {
            dimensions.push_back(a.shape(i));
        }
        return dimensions;
    }

    //static constexpr bool show_writeable = is_eigen_dense_map<Type>::value && is_eigen_mutable_map<Type>::value;
    //static constexpr bool show_order = is_eigen_dense_map<Type>::value;
    //static constexpr bool show_c_contiguous = show_order && requires_row_major;
    //static constexpr bool show_f_contiguous = !show_c_contiguous && show_order && requires_col_major;

    static constexpr auto descriptor =
        _("numpy.ndarray[") + npy_format_descriptor<Scalar>::name +
        _("[")  + tensor_dimensions<dimensions>::text +
        _("]") +
        // For a reference type (e.g. Ref<MatrixXd>) we have other constraints that might need to be
        // satisfied: writeable=True (for a mutable reference), and, depending on the map's stride
        // options, possibly f_contiguous or c_contiguous.  We include them in the descriptor output
        // to provide some hint as to why a TypeError is occurring (otherwise it can be confusing to
        // see that a function accepts a 'numpy.ndarray[float64[3,2]]' and an error message that you
        // *gave* a numpy.ndarray of the right type and dimensions.
        //_<show_writeable>(", flags.writeable", "") +
        //_<show_c_contiguous>(", flags.c_contiguous", "") +
        //_<show_f_contiguous>(", flags.f_contiguous", "") +
        _("]");
};

// Casts an Eigen type to numpy array.  If given a base, the numpy array references the src data,
// otherwise it'll make a copy.  writeable lets you turn off the writeable flag for the array.
template <typename props>
handle eigen_array_cast(typename props::Type const &src, handle base = handle(),
                        bool writeable = true) {
  constexpr ssize_t elem_size = sizeof(typename props::Scalar);
  array a{src.Dimensions, src.data(), base};

  //if (!writeable)
  //  array_proxy(a.ptr())->flags &= ~detail::npy_api::NPY_ARRAY_WRITEABLE_;

  return a.release();
}

// Takes an lvalue ref to some Eigen type and a (python) base object, creating a numpy array that
// reference the Eigen object's data with `base` as the python-registered base class (if omitted,
// the base will be set to None, and lifetime management is up to the caller).  The numpy array is
// non-writeable if the given type is const.
template <typename props, typename Type>
handle eigen_ref_array(Type &src, handle parent = none()) {
    // none here is to get past array's should-we-copy detection, which currently always
    // copies when there is no base.  Setting the base to None should be harmless.
    return eigen_array_cast<props>(src, parent, !std::is_const<Type>::value);
}

// Takes a pointer to some dense, plain Eigen type, builds a capsule around it, then returns a numpy
// array that references the encapsulated data with a python-side reference to the capsule to tie
// its destruction to that of any dependent python objects.  Const-ness is determined by whether or
// not the Type of the pointer given is const.
template <typename props, typename Type>
handle eigen_encapsulate(Type *src) {
    capsule base(src, [](void *o) { delete static_cast<Type *>(o); });
    return eigen_ref_array<props>(*src, base);
}

// Type caster for regular, dense matrix types (e.g. MatrixXd), but not maps/refs/etc. of dense
// types.
template<typename Type>
struct type_caster<Type, enable_if_t<is_eigen_tensor<Type>::value>> {
    using Scalar = typename Type::Scalar;
    using props = EigenTensorProps<Type>;

    bool load(handle src, bool convert) {
        // If we're in no-convert mode, only load if given an array of the correct type
        if (!convert && !isinstance<array_t<Scalar>>(src))
            return false;

        // Coerce into an array, but don't do type conversion yet; the copy below handles it.
        auto buf = array::ensure(src);

        if (!buf)
            return false;

        auto dims = buf.ndim();
        if (dims < 1 || dims > 2)
            return false;

        auto fits = props::conformable(buf);
        if (!fits)
            return false;

        // Allocate the new type, then build a numpy reference into it
        value = Type(fits.rows, fits.cols);
        auto ref = reinterpret_steal<array>(eigen_ref_array<props>(value));
        if (dims == 1) ref = ref.squeeze();
        else if (ref.ndim() == 1) buf = buf.squeeze();

        int result = detail::npy_api::get().PyArray_CopyInto_(ref.ptr(), buf.ptr());

        if (result < 0) { // Copy failed!
            PyErr_Clear();
            return false;
        }

        return true;
    }

private:

    // Cast implementation
    template <typename CType>
    static handle cast_impl(CType *src, return_value_policy policy, handle parent) {
        switch (policy) {
            case return_value_policy::take_ownership:
            case return_value_policy::automatic:
                return eigen_encapsulate<props>(src);
            case return_value_policy::move:
                return eigen_encapsulate<props>(new CType(std::move(*src)));
            case return_value_policy::copy:
                return eigen_array_cast<props>(*src);
            case return_value_policy::reference:
            case return_value_policy::automatic_reference:
                return eigen_ref_array<props>(*src);
            case return_value_policy::reference_internal:
                return eigen_ref_array<props>(*src, parent);
            default:
                throw cast_error("unhandled return_value_policy: should not happen!");
        };
    }

public:

    // Normal returned non-reference, non-const value:
    static handle cast(Type &&src, return_value_policy /* policy */, handle parent) {
        return cast_impl(&src, return_value_policy::move, parent);
    }
    // If you return a non-reference const, we mark the numpy array readonly:
    static handle cast(const Type &&src, return_value_policy /* policy */, handle parent) {
        return cast_impl(&src, return_value_policy::move, parent);
    }
    // lvalue reference return; default (automatic) becomes copy
    static handle cast(Type &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::copy;
        return cast_impl(&src, policy, parent);
    }
    // const lvalue reference return; default (automatic) becomes copy
    static handle cast(const Type &src, return_value_policy policy, handle parent) {
        if (policy == return_value_policy::automatic || policy == return_value_policy::automatic_reference)
            policy = return_value_policy::copy;
        return cast(&src, policy, parent);
    }
    // non-const pointer return
    static handle cast(Type *src, return_value_policy policy, handle parent) {
        return cast_impl(src, policy, parent);
    }
    // const pointer return
    static handle cast(const Type *src, return_value_policy policy, handle parent) {
        return cast_impl(src, policy, parent);
    }

    static constexpr auto name = props::descriptor;

    operator Type*() { return &value; }
    operator Type&() { return value; }
    operator Type&&() && { return std::move(value); }
    template <typename T> using cast_op_type = movable_cast_op_type<T>;

private:
    Type value;
};

NAMESPACE_END(detail)
NAMESPACE_END(PYBIND11_NAMESPACE)

#if defined(__GNUG__) || defined(__clang__)
#  pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#  pragma warning(pop)
#endif
