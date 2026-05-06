// FITS read/write helpers, implemented against the cfitsio C API.
//
// The whole file is gated on LINA_USE_CFITSIO. When the library was
// configured without cfitsio (-DLINA_USE_CFITSIO=OFF), the throwing
// stub implementations below are compiled instead so callers always
// link successfully.
#include "lina/utils.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

#ifdef LINA_USE_CFITSIO
#include <fitsio.h>
#endif

namespace lina {

#ifdef LINA_USE_CFITSIO

namespace {

// Throw a descriptive error containing the cfitsio status text.
[[noreturn]] void throw_cfitsio(int status, const char* context) {
    char err_text[FLEN_STATUS] = {0};
    fits_get_errstatus(status, err_text);
    throw std::runtime_error(std::string("cfitsio: ") + context + ": " + err_text);
}

// If the destination file already exists, cfitsio refuses to overwrite
// unless the path is prefixed with '!'. We honour the `overwrite` flag
// to match astropy's `overwrite=True` semantics.
std::string overwrite_path(const std::string& fpath, bool overwrite) {
    if (overwrite) {
        // Be sure not to double-prefix.
        if (!fpath.empty() && fpath.front() == '!') return fpath;
        return std::string("!") + fpath;
    }
    return fpath;
}

// Write the user-supplied header as additional keys on the primary HDU.
// We call fits_write_key_str so cfitsio quotes string values; numeric
// values pass through unchanged in their text form which is fine for
// metadata round-trips.
void write_user_header(fitsfile* fptr, const FitsHeader& header) {
    int status = 0;
    for (const auto& kv : header) {
        // Reserved keys would corrupt the HDU; skip them.
        const std::string& key = kv.first;
        if (key == "SIMPLE" || key == "BITPIX" || key == "NAXIS" ||
            key == "NAXIS1" || key == "NAXIS2" || key == "EXTEND") continue;
        // cfitsio takes mutable char*; copy into a small buffer.
        char k[FLEN_KEYWORD]; char v[FLEN_VALUE];
        std::strncpy(k, key.c_str(), FLEN_KEYWORD - 1); k[FLEN_KEYWORD - 1] = 0;
        std::strncpy(v, kv.second.c_str(), FLEN_VALUE - 1); v[FLEN_VALUE - 1] = 0;
        fits_write_key_str(fptr, k, v, /*comment=*/nullptr, &status);
        if (status) throw_cfitsio(status, "write_key_str");
    }
}

// Generic save: write a contiguous data buffer into a 2D primary HDU.
// `bitpix` and `datatype` together describe the on-disk and in-memory
// representation respectively (e.g. {DOUBLE_IMG, TDOUBLE}).
template <typename T>
void save_fits_impl(const std::string& fpath,
                    const Array2D<T>& data,
                    const FitsHeader& header,
                    bool overwrite,
                    int bitpix,
                    int datatype) {
    int status = 0;
    fitsfile* fptr = nullptr;
    const std::string path = overwrite_path(fpath, overwrite);
    fits_create_file(&fptr, path.c_str(), &status);
    if (status) throw_cfitsio(status, "fits_create_file");

    // FITS axes are FITS-Fortran-order: NAXIS1 fastest-varying. Our
    // Array2D row-major layout means each row is contiguous, so cols=
    // NAXIS1, rows=NAXIS2.
    long naxes[2] = { static_cast<long>(data.cols()),
                      static_cast<long>(data.rows()) };
    fits_create_img(fptr, bitpix, 2, naxes, &status);
    if (status) {
        int closing = 0; fits_close_file(fptr, &closing);
        throw_cfitsio(status, "fits_create_img");
    }

    write_user_header(fptr, header);

    // First-pixel coordinate is 1-based per FITS convention.
    long fpixel[2] = {1, 1};
    const long nelements = static_cast<long>(data.size());
    // const_cast: cfitsio's signature takes void* but doesn't write through it.
    fits_write_pix(fptr, datatype, fpixel, nelements,
                   const_cast<T*>(data.data()), &status);
    if (status) {
        int closing = 0; fits_close_file(fptr, &closing);
        throw_cfitsio(status, "fits_write_pix");
    }

    fits_close_file(fptr, &status);
    if (status) throw_cfitsio(status, "fits_close_file");
}

// Load a 2D primary HDU as Array2D<T>. cfitsio handles on-disk to
// in-memory dtype conversion, so a file written with BITPIX=16 (int16)
// can still be loaded as double.
template <typename T>
Array2D<T> load_fits_impl(const std::string& fpath, int datatype) {
    int status = 0;
    fitsfile* fptr = nullptr;
    fits_open_image(&fptr, fpath.c_str(), READONLY, &status);
    if (status) throw_cfitsio(status, "fits_open_image");

    int naxis = 0;
    fits_get_img_dim(fptr, &naxis, &status);
    if (status) {
        int closing = 0; fits_close_file(fptr, &closing);
        throw_cfitsio(status, "fits_get_img_dim");
    }
    if (naxis != 2) {
        int closing = 0; fits_close_file(fptr, &closing);
        throw std::runtime_error("lina::load_fits: expected 2D image, got NAXIS=" +
                                 std::to_string(naxis));
    }

    long naxes[2] = {0, 0};
    fits_get_img_size(fptr, 2, naxes, &status);
    if (status) {
        int closing = 0; fits_close_file(fptr, &closing);
        throw_cfitsio(status, "fits_get_img_size");
    }

    // Map FITS (NAXIS1=cols, NAXIS2=rows) back to our Array2D row-major.
    Array2D<T> out(static_cast<std::size_t>(naxes[1]),
                   static_cast<std::size_t>(naxes[0]),
                   T());
    long fpixel[2] = {1, 1};
    int anynul = 0;
    fits_read_pix(fptr, datatype, fpixel,
                  static_cast<long>(out.size()),
                  /*nulval=*/nullptr, out.data(), &anynul, &status);
    if (status) {
        int closing = 0; fits_close_file(fptr, &closing);
        throw_cfitsio(status, "fits_read_pix");
    }
    fits_close_file(fptr, &status);
    if (status) throw_cfitsio(status, "fits_close_file");
    return out;
}

// Walk all keys in the primary HDU and return them as (key, value) pairs.
FitsHeader read_header_impl(fitsfile* fptr) {
    int status = 0;
    int nkeys = 0;
    fits_get_hdrspace(fptr, &nkeys, /*morekeys=*/nullptr, &status);
    if (status) throw_cfitsio(status, "fits_get_hdrspace");

    FitsHeader hdr;
    hdr.reserve(nkeys);
    for (int i = 1; i <= nkeys; ++i) {
        char keyname[FLEN_KEYWORD] = {0};
        char keyvalue[FLEN_VALUE] = {0};
        char keycomment[FLEN_COMMENT] = {0};
        fits_read_keyn(fptr, i, keyname, keyvalue, keycomment, &status);
        if (status) throw_cfitsio(status, "fits_read_keyn");
        hdr.emplace_back(std::string(keyname), std::string(keyvalue));
    }
    return hdr;
}

} // namespace

void save_fits(const std::string& fpath,
               const Array2D<double>& data,
               const FitsHeader& header,
               bool overwrite) {
    save_fits_impl(fpath, data, header, overwrite, DOUBLE_IMG, TDOUBLE);
}

void save_fits(const std::string& fpath,
               const Array2D<float>& data,
               const FitsHeader& header,
               bool overwrite) {
    save_fits_impl(fpath, data, header, overwrite, FLOAT_IMG, TFLOAT);
}

void save_fits(const std::string& fpath,
               const Array2D<std::int32_t>& data,
               const FitsHeader& header,
               bool overwrite) {
    save_fits_impl(fpath, data, header, overwrite, LONG_IMG, TINT);
}

void save_fits(const std::string& fpath,
               const Array2D<std::uint8_t>& data,
               const FitsHeader& header,
               bool overwrite) {
    save_fits_impl(fpath, data, header, overwrite, BYTE_IMG, TBYTE);
}

Array2D<double> load_fits_double(const std::string& fpath) {
    return load_fits_impl<double>(fpath, TDOUBLE);
}

Array2D<float> load_fits_float(const std::string& fpath) {
    return load_fits_impl<float>(fpath, TFLOAT);
}

std::pair<Array2D<double>, FitsHeader> load_fits_with_header(const std::string& fpath) {
    int status = 0;
    fitsfile* fptr = nullptr;
    fits_open_image(&fptr, fpath.c_str(), READONLY, &status);
    if (status) throw_cfitsio(status, "fits_open_image");

    FitsHeader hdr;
    try {
        hdr = read_header_impl(fptr);
    } catch (...) {
        int closing = 0; fits_close_file(fptr, &closing);
        throw;
    }

    int naxis = 0;
    fits_get_img_dim(fptr, &naxis, &status);
    if (status) { int c=0; fits_close_file(fptr,&c); throw_cfitsio(status, "fits_get_img_dim"); }
    if (naxis != 2) {
        int c = 0; fits_close_file(fptr, &c);
        throw std::runtime_error("lina::load_fits_with_header: expected 2D image");
    }

    long naxes[2] = {0, 0};
    fits_get_img_size(fptr, 2, naxes, &status);
    if (status) { int c=0; fits_close_file(fptr,&c); throw_cfitsio(status, "fits_get_img_size"); }

    Array2D<double> out(static_cast<std::size_t>(naxes[1]),
                        static_cast<std::size_t>(naxes[0]),
                        0.0);
    long fpixel[2] = {1, 1};
    int anynul = 0;
    fits_read_pix(fptr, TDOUBLE, fpixel, static_cast<long>(out.size()),
                  nullptr, out.data(), &anynul, &status);
    if (status) { int c=0; fits_close_file(fptr,&c); throw_cfitsio(status, "fits_read_pix"); }
    fits_close_file(fptr, &status);
    if (status) throw_cfitsio(status, "fits_close_file");
    return {std::move(out), std::move(hdr)};
}

bool fits_available() { return true; }

#else // LINA_USE_CFITSIO not defined: provide throwing stubs.

[[noreturn]] static void no_fits() {
    throw std::runtime_error(
        "lina was built without cfitsio support; "
        "rebuild with -DLINA_USE_CFITSIO=ON.");
}

void save_fits(const std::string&, const Array2D<double>&, const FitsHeader&, bool) { no_fits(); }
void save_fits(const std::string&, const Array2D<float>&,  const FitsHeader&, bool) { no_fits(); }
void save_fits(const std::string&, const Array2D<std::int32_t>&,  const FitsHeader&, bool) { no_fits(); }
void save_fits(const std::string&, const Array2D<std::uint8_t>&,  const FitsHeader&, bool) { no_fits(); }
Array2D<double> load_fits_double(const std::string&) { no_fits(); }
Array2D<float>  load_fits_float(const std::string&)  { no_fits(); }
std::pair<Array2D<double>, FitsHeader> load_fits_with_header(const std::string&) { no_fits(); }
bool fits_available() { return false; }

#endif // LINA_USE_CFITSIO

} // namespace lina
