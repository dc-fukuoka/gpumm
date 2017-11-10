#define myassert(expr, word) \
  if ((expr) .eq. .false.) then; \
     write(6,*) word, ":", __FILE__, ":", __LINE__; \
     call tracebackqq(); \
     stop; \
  end if

module subs
contains
  subroutine gen_rand(seed, min, max, size, arr)
    use mkl_vsl_type
    use mkl_vsl
    implicit none
    integer,intent(in) :: seed
    real(8),intent(in) :: min, max
    integer,intent(in) :: size
    real(8),dimension(size, size),intent(out) :: arr
    integer :: ierr
    integer :: brng, method
    type(vsl_stream_state) :: stream

    brng   = vsl_brng_mt19937
    method = vsl_rng_method_uniform_std_accurate
    
    ierr = vslnewstream(stream, brng, seed)
    myassert(ierr.eq.vsl_error_ok, "vslnewstream()")
    ierr = vdrnguniform(method, stream, size, arr, min, max)
    myassert(ierr.eq.vsl_error_ok, "vdrnguniform()")
    ierr = vsldeletestream(stream)
    myassert(ierr.eq.vsl_error_ok, "vsldeletestream()")
    
  end subroutine gen_rand

  subroutine mydgemm(size, a, b, c)
    implicit none
    integer,intent(in) :: size
    real(8),dimension(size, size),intent(in) :: a, b
    real(8),dimension(size, size),intent(inout) :: c
    integer :: i, j, k

    !$omp parallel do private(i, j, k)
    do j = 1, size
       do k = 1, size
          do i = 1, size
             c(i, j) = c(i, j) + a(i, k)*b(k, j)
          end do
       end do
    end do

  end subroutine mydgemm

  function calc_trace(size, c) result(trace)
    implicit none
    integer,intent(in) :: size
    real(8),dimension(size, size),intent(in) :: c
    real(8) :: trace
    integer :: i

    trace = 0.0d0
    !$omp parallel do reduction(+:trace)
    do i = 1, size
       trace = trace + c(i, i)
    end do
    
  end function calc_trace

  subroutine transpose(size, c)
    implicit none
    integer,intent(in) :: size
    real(8),dimension(size, size),intent(inout) :: c
    real(8),dimension(size, size) :: c2
    integer :: i, j

    !$omp parallel private(i, j)
    !$omp do
    do j = 1, size
       do i = 1, size
          c2(j, i) = c(i, j)
       end do
    end do
    !$omp end do
    !$omp do
    do j = 1, size
       do i = 1, size
          c(i, j) = c2(i, j)
       end do
    end do
    !$omp end do
    !$omp end parallel
    
  end subroutine transpose
end module subs

program main
  use subs
  implicit none
  integer :: size
  character(len=32) :: argv1
  real(8),allocatable,dimension(:,:) :: a, b, c
  real(8) :: dclock, time, t0
  real(8) :: trace

  if (iargc() .eq. 0) then
     size = 1024
  else
     call getarg(1, argv1)
     read(argv1, *) size
  end if

  write(6, '("size: ", i0)') size
  allocate(a(size, size), b(size, size), c(size, size))
  c(:, :) = 0.0d0
  
  call gen_rand(5555, -1.0d0, 1.0d0, size*size, a)
  call gen_rand(7777, -1.0d0, 1.0d0, size*size, b)
  call transpose(size, a)
  call transpose(size, b)

  t0 = dclock()
  call mydgemm(size, a, b, c)
  time = dclock() - t0

  trace = calc_trace(size, c)
  call transpose(size, c)
  write(6, '("time[s]: ", f9.6)') time
  write(6, '("trace: ", 1pe22.15)') trace

  open(100, file="C_f", form="unformatted", access="stream")
  write(100) c
  close(100)
  
  deallocate(a, b, c)
  stop
end program main
