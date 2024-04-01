
import { useSelector } from 'react-redux';
import { Navigate, Outlet } from 'react-router-dom';
import { RootState } from '../store/store';

export const ProtectedRoute = () => {
  const accessToken = useSelector((state:RootState) => state.accessToken.value)
  return accessToken ? <Outlet /> : <Navigate to="/login" replace />;
};