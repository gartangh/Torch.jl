function get_error()
  err = cglobal((:myerr, :libdoeye_caml), Cstring) |> unsafe_load
  unsafe_string(err)
end

function flush_error()
  ccall((:flush_error, :libdoeye_caml), Cvoid, ())
end

macro runtime_error_check(ex)
  quote
    x = $ex
    # @show x
    if x == 1
      cs = get_error()
      flush_error()
      throw(cs)
    end
  end |> esc
end
