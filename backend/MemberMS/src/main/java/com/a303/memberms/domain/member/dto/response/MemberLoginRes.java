package com.a303.memberms.domain.member.dto.response;


import lombok.Builder;
import lombok.Getter;


@Builder
public record MemberLoginRes(
    @Getter
    int memberId,
    @Getter
    String token
) {

}
