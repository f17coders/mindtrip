
import { useSelector } from 'react-redux';
import { Navigate, Outlet } from 'react-router-dom';
import { RootState } from '../store/store';

export const ProtectedRoute = () => {
  const memberId = useSelector((state:RootState) => state.member.memberId)
  return memberId ? <Outlet /> : <Navigate to="/login" replace />;
};