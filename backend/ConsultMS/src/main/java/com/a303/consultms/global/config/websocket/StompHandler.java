//package com.a303.consultms.global.config.websocket;
//
//
//import com.a303.consultms.global.security.exception.AccessTokenException;
//import io.jsonwebtoken.ExpiredJwtException;
//import io.jsonwebtoken.Jwts;
//import io.jsonwebtoken.MalformedJwtException;
//import io.jsonwebtoken.SignatureException;
//import java.util.Map;
//import java.util.Objects;
//import lombok.RequiredArgsConstructor;
//import lombok.extern.slf4j.Slf4j;
//import org.springframework.beans.factory.annotation.Value;
//import org.springframework.core.Ordered;
//import org.springframework.core.annotation.Order;
//import org.springframework.messaging.Message;
//import org.springframework.messaging.MessageChannel;
//import org.springframework.messaging.simp.stomp.StompCommand;
//import org.springframework.messaging.simp.stomp.StompHeaderAccessor;
//import org.springframework.messaging.support.ChannelInterceptor;
//import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
//import org.springframework.security.core.Authentication;
//import org.springframework.security.core.context.SecurityContextHolder;
//import org.springframework.stereotype.Component;
//
//@Component
//@RequiredArgsConstructor
//@Order(Ordered.HIGHEST_PRECEDENCE + 99)
//@Slf4j
//public class StompHandler implements ChannelInterceptor {
//
//    @Value(value = "${key.jwt.secret}")
//    private String jwtKey;
//
//    @Override
//    public Message<?> preSend(Message<?> message, MessageChannel channel) {
//        StompHeaderAccessor accessor = StompHeaderAccessor.wrap(message);
//
//        if (StompCommand.CONNECT.equals(accessor.getCommand())) {
//            Map<String, Object> payload = validateAccessToken(
//                Objects.requireNonNull(accessor.getFirstNativeHeader("Authorization")));
//
//            System.out.println(payload);
//            String memberId = (String) payload.get("memberId");
//
//            // 인증 객체 생성 (UsernamePasswordAuthenticationToken 사용 예)
//            Authentication authenticationToken = new UsernamePasswordAuthenticationToken(memberId, null);
//            SecurityContextHolder.getContext().setAuthentication(authenticationToken);
//        }
//        return message;
//    }
//
//    private Map<String, Object> validateAccessToken(String headStr) {
//        if (Objects.isNull(headStr)) {
//            throw new AccessTokenException(AccessTokenException.TOKEN_ERROR.UNACCEPT);
//        }
//        String tokenType = headStr.substring(0, 6);     // BEARER
//        String tokenStr = headStr.substring(7);
//
//        if (!tokenType.equalsIgnoreCase("BEARER")) {
//            throw new AccessTokenException(AccessTokenException.TOKEN_ERROR.BADTYPE);
//        }
//
//        try {
//            return validateToken(tokenStr);
//        } catch (MalformedJwtException malformedJwtException) {
//            throw new AccessTokenException(AccessTokenException.TOKEN_ERROR.MALFORM);
//        } catch (SignatureException signatureException) {
//            throw new AccessTokenException(AccessTokenException.TOKEN_ERROR.BADSIGN);
//        } catch (ExpiredJwtException expiredJwtException) {
//            throw new AccessTokenException(AccessTokenException.TOKEN_ERROR.EXPIRED);
//        }
//    }
//    public Map<String, Object> validateToken(String token) {    // 주어진 JWT 토큰이 유효한지 검증하고, 토큰에서 정보를 추출하여 반환
//        Map<String, Object> claim = null;
//
//        log.info("[JWTUtil.java - validateToken(), token = " + token);
//
//        claim = Jwts.parser()
//            .setSigningKey(jwtKey.getBytes()) // Set Key
//            .parseClaimsJws(token) // 파싱 및 검증, 실패 시 에러
//            .getBody();
//
//        return claim;   // 성공하면 토큰의 페이로드를 반환
//    }
//}
